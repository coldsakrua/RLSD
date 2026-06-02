from __future__ import annotations

import importlib
import functools
import json
import logging
import os
import weakref
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch

logger = logging.getLogger(__name__)

_CONTROLLER: TeacherEMAController | None = None
_TRAINER_REF: weakref.ReferenceType[Any] | None = None
_PATCHED = False


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _cfg_get(config: Any, path: str, default: Any = None) -> Any:
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _parse_target_modules(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = []
    for chunk in text.replace(",", " ").split():
        item = chunk.strip().strip("'\"")
        if item:
            parts.append(item)
    return parts


class TeacherEMAController:
    """Maintain an EMA shadow of student LoRA weights and push it to teacher vLLM."""

    def __init__(
        self,
        *,
        enabled: bool,
        decay: float,
        update_interval_steps: int,
        lora_name: str,
        adapter_dir: Path,
        lora_r: int,
        lora_alpha: int,
        target_modules: list[str],
        base_model_path: str,
    ) -> None:
        self.enabled = enabled
        self.decay = float(decay)
        self.update_interval_steps = max(1, int(update_interval_steps))
        self.lora_name = lora_name
        self.adapter_dir = adapter_dir
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.target_modules = target_modules
        self.base_model_path = base_model_path
        self._ema_state: dict[str, torch.Tensor] | None = None
        self._last_sync_step = -1

    @classmethod
    def from_trainer(cls, trainer: Any) -> TeacherEMAController | None:
        config = trainer.config
        enabled = bool(_cfg_get(config, "algorithm.rlsd.teacher_ema_enabled", False))
        if not enabled:
            return None
        if _env_truthy("RLSD_INTERNAL_TEACHER_LOGPROB", False):
            return None
        output_dir = Path(str(_cfg_get(config, "trainer.default_local_dir", ".")))
        adapter_dir = output_dir / str(
            _cfg_get(config, "algorithm.rlsd.teacher_ema_adapter_dir", "teacher_ema_adapter")
        )
        return cls(
            enabled=True,
            decay=float(_cfg_get(config, "algorithm.rlsd.teacher_ema_decay", 0.99)),
            update_interval_steps=int(
                _cfg_get(config, "algorithm.rlsd.teacher_ema_update_interval_steps", 1)
            ),
            lora_name=str(_cfg_get(config, "algorithm.rlsd.teacher_ema_lora_name", "teacher_ema")),
            adapter_dir=adapter_dir,
            lora_r=int(_cfg_get(config, "actor_rollout_ref.model.lora_rank", 0) or 0),
            lora_alpha=int(_cfg_get(config, "actor_rollout_ref.model.lora_alpha", 0) or 0),
            target_modules=_parse_target_modules(_cfg_get(config, "actor_rollout_ref.model.target_modules", [])),
            base_model_path=str(_cfg_get(config, "actor_rollout_ref.model.path", "")),
        )

    def should_update(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        step = int(global_step or 0)
        if step <= 0:
            return False
        if step == self._last_sync_step:
            return False
        return step % self.update_interval_steps == 0

    def update_from_student(self, student_params: dict[str, torch.Tensor]) -> None:
        if not student_params:
            raise RuntimeError("Teacher EMA sync received empty student LoRA parameters.")
        decay = self.decay
        with torch.no_grad():
            if self._ema_state is None:
                self._ema_state = {name: tensor.detach().cpu().clone() for name, tensor in student_params.items()}
                return
            for name, tensor in student_params.items():
                student_cpu = tensor.detach().cpu()
                if name in self._ema_state:
                    self._ema_state[name].mul_(decay).add_(student_cpu, alpha=1.0 - decay)
                else:
                    self._ema_state[name] = student_cpu.clone()

    def save_adapter(self) -> Path:
        if not self._ema_state:
            raise RuntimeError("Teacher EMA state is empty; sync student LoRA before saving.")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_config = {
            "peft_type": "LORA",
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": self.base_model_path,
        }
        with open(self.adapter_dir / "adapter_config.json", "w", encoding="utf-8") as handle:
            json.dump(adapter_config, handle, indent=2)
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to export teacher EMA LoRA adapters.") from exc
        save_file(self._ema_state, str(self.adapter_dir / "adapter_model.safetensors"))
        return self.adapter_dir

    def _server_url(self, server_address: str, path: str) -> str:
        address = server_address.strip()
        if address.startswith("http://") or address.startswith("https://"):
            return address.rstrip("/") + path
        return f"http://{address.rstrip('/')}{path}"

    def _post_json(self, url: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=120) as response:
            response.read()

    def sync_to_teacher_manager(self, teacher_model_manager: Any) -> None:
        adapter_path = self.save_adapter()
        managers = getattr(teacher_model_manager, "teacher_model_managers", None)
        if not managers:
            manager = teacher_model_manager
            addresses = getattr(manager, "server_addresses", [])
            for address in addresses:
                self._load_lora_http(address, adapter_path)
            return
        for manager in managers.values():
            for address in getattr(manager, "server_addresses", []):
                self._load_lora_http(address, adapter_path)

    def _load_lora_http(self, server_address: str, adapter_path: Path) -> None:
        try:
            utils_mod = importlib.import_module("verl.workers.rollout.vllm_rollout.utils")
            lora_name = getattr(utils_mod, "VLLM_LORA_NAME", self.lora_name)
        except Exception:
            lora_name = self.lora_name
        url = self._server_url(server_address, "/v1/load_lora_adapter")
        payload = {
            "lora_name": lora_name,
            "lora_path": str(adapter_path.resolve()),
            "load_inplace": True,
        }
        try:
            self._post_json(url, payload)
            logger.info("Loaded teacher EMA LoRA %s from %s via %s", self.lora_name, adapter_path, url)
        except urllib_error.URLError as exc:
            logger.warning("Failed to load teacher EMA LoRA via %s: %s", url, exc)

    def maybe_sync(self, trainer: Any) -> None:
        if not self.enabled:
            return
        global_step = int(getattr(trainer, "global_steps", 0) or 0)
        if not self.should_update(global_step):
            return
        student_params = collect_student_lora_params(trainer)
        if not student_params:
            logger.warning("Teacher EMA enabled but student LoRA export returned nothing at step %s.", global_step)
            return
        self.update_from_student(student_params)
        teacher_manager = getattr(trainer, "teacher_model_manager", None)
        if teacher_manager is None:
            logger.warning("Teacher EMA updated in memory but teacher_model_manager is missing.")
        else:
            self.sync_to_teacher_manager(teacher_manager)
        os.environ["RLSD_TEACHER_EMA_ADAPTER_PATH"] = str(self.adapter_dir.resolve())
        self._last_sync_step = global_step


def collect_student_lora_params(trainer: Any) -> dict[str, torch.Tensor]:
    wg = getattr(trainer, "actor_rollout_wg", None)
    if wg is None:
        return {}
    method_names = (
        "export_lora_state_dict_for_teacher_ema",
        "collect_lora_params",
        "export_lora_state_dict",
        "get_lora_state_dict",
    )
    for method_name in method_names:
        try:
            if hasattr(wg, method_name):
                result = getattr(wg, method_name)()
            elif hasattr(wg, "execute_rank_zero_sync"):
                result = wg.execute_rank_zero_sync(method_name)
            else:
                continue
            if isinstance(result, dict) and result:
                return {k: v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for k, v in result.items()}
        except Exception:
            continue
    return {}


def register_teacher_ema_controller(controller: TeacherEMAController | None, trainer: Any) -> None:
    global _CONTROLLER, _TRAINER_REF
    _CONTROLLER = controller
    _TRAINER_REF = weakref.ref(trainer) if trainer is not None else None


def _maybe_sync_from_registered_trainer() -> None:
    if _CONTROLLER is None or _TRAINER_REF is None:
        return
    trainer = _TRAINER_REF()
    if trainer is None:
        return
    _CONTROLLER.maybe_sync(trainer)


def _patch_actor_worker_export() -> None:
    candidate_modules = (
        "verl.workers.fsdp_workers",
        "verl.workers.megatron_workers",
        "verl.workers.roles.actor_rollout_ref",
    )
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        worker_cls = None
        for attr in ("ActorRolloutRefWorker", "ActorWorker", "FSDPWorker"):
            worker_cls = getattr(module, attr, None)
            if worker_cls is not None:
                break
        if worker_cls is None or hasattr(worker_cls, "export_lora_state_dict_for_teacher_ema"):
            continue

        def export_lora_state_dict_for_teacher_ema(self):  # type: ignore[no-untyped-def]
            from verl.utils.fsdp_utils import collect_lora_params

            module_candidates = (
                "actor_module_fsdp",
                "actor_module",
                "module",
                "model",
            )
            fsdp_module = None
            for name in module_candidates:
                candidate = getattr(self, name, None)
                if candidate is not None:
                    fsdp_module = candidate
                    break
            if fsdp_module is None:
                return {}
            rollout_cfg = getattr(self, "config", None)
            layered_summon = False
            base_sync_done = True
            if rollout_cfg is not None:
                layered_summon = bool(
                    _cfg_get(rollout_cfg, "actor_rollout_ref.rollout.layered_summon", False)
                )
                base_sync_done = bool(getattr(self, "base_sync_done", True))
            params = collect_lora_params(fsdp_module, layered_summon, base_sync_done)
            return {name: tensor.detach().cpu() for name, tensor in params.items()}

        worker_cls.export_lora_state_dict_for_teacher_ema = export_lora_state_dict_for_teacher_ema
        logger.info("Patched %s.export_lora_state_dict_for_teacher_ema", worker_cls.__name__)
        return


def _get_item(container: Any, key: str) -> Any:
    try:
        return container[key]
    except Exception:
        pass
    try:
        return container.get(key)
    except Exception:
        return None


def _set_item(container: Any, key: str, value: Any) -> None:
    try:
        container[key] = value
        return
    except Exception:
        pass
    batch = getattr(container, "batch", None)
    if batch is not None:
        batch[key] = value


def _candidate_model_roots(worker: Any) -> list[Any]:
    roots: list[Any] = []
    actor = getattr(worker, "actor", None)
    engine = getattr(actor, "engine", None)
    for obj in (worker, actor, engine):
        if obj is not None:
            roots.append(obj)
    attr_names = (
        "actor_module_fsdp",
        "actor_module",
        "fsdp_module",
        "model_module",
        "model",
        "module",
        "_model",
    )
    for obj in list(roots):
        for name in attr_names:
            candidate = getattr(obj, name, None)
            if candidate is not None:
                roots.append(candidate)
    deduped: list[Any] = []
    seen: set[int] = set()
    for root in roots:
        ident = id(root)
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(root)
    return deduped


def _iter_lora_params(module: Any) -> list[tuple[str, torch.nn.Parameter]]:
    try:
        named_parameters = module.named_parameters
    except AttributeError:
        return []
    teacher_adapter = os.environ.get("RLSD_TEACHER_EMA_LORA_NAME", "teacher_ema")
    params: list[tuple[str, torch.nn.Parameter]] = []
    try:
        iterator = named_parameters()
    except Exception:
        return []
    for name, param in iterator:
        lname = str(name).lower()
        if "lora_" not in lname:
            continue
        if teacher_adapter and f".{teacher_adapter.lower()}." in lname:
            continue
        if not isinstance(param, torch.nn.Parameter):
            continue
        params.append((str(name), param))
    return params


def _find_lora_module(worker: Any) -> tuple[Any | None, list[tuple[str, torch.nn.Parameter]]]:
    for root in _candidate_model_roots(worker):
        params = _iter_lora_params(root)
        if params:
            return root, params
    return None, []


def _teacher_ema_decay() -> float:
    raw = os.environ.get("RLSD_TEACHER_EMA_DECAY", "0.995")
    try:
        decay = float(raw)
    except ValueError:
        decay = 0.995
    return min(max(decay, 0.0), 1.0)


def _ensure_internal_ema_state(worker: Any, params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    state = getattr(worker, "_rlsd_internal_teacher_ema_state", None)
    if not isinstance(state, dict) or not state:
        state = {name: param.detach().clone() for name, param in params}
        setattr(worker, "_rlsd_internal_teacher_ema_state", state)
    return state


def _update_internal_teacher_ema(worker: Any) -> bool:
    if not _env_truthy("RLSD_INTERNAL_TEACHER_LOGPROB", False):
        return False
    _, params = _find_lora_module(worker)
    if not params:
        if not getattr(worker, "_rlsd_internal_teacher_warned", False):
            logger.warning("Internal EMA teacher is enabled but no LoRA parameters were found on actor worker.")
            setattr(worker, "_rlsd_internal_teacher_warned", True)
        return False
    state = _ensure_internal_ema_state(worker, params)
    decay = _teacher_ema_decay()
    with torch.no_grad():
        for name, param in params:
            current = param.detach()
            previous = state.get(name)
            if previous is None or tuple(previous.shape) != tuple(current.shape):
                state[name] = current.clone()
                continue
            if previous.device != current.device or previous.dtype != current.dtype:
                previous = previous.to(device=current.device, dtype=current.dtype)
            previous.mul_(decay).add_(current, alpha=1.0 - decay)
            state[name] = previous
    return True


def _teacher_tensor_from_output(output: Any) -> torch.Tensor | None:
    for key in ("old_log_probs", "old_log_prob", "log_probs", "log_prob"):
        value = _get_item(output, key)
        if isinstance(value, torch.Tensor):
            return value.detach()
    return None


def _compute_internal_teacher_logprobs(worker: Any, data: Any, compute_fn: Any = None) -> torch.Tensor | None:
    _, params = _find_lora_module(worker)
    if not params:
        return None
    state = _ensure_internal_ema_state(worker, params)
    teacher_values: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for name, param in params:
        teacher_value = state.get(name)
        if teacher_value is None or tuple(teacher_value.shape) != tuple(param.shape):
            return None
        teacher_values.append((param, teacher_value))
    backups: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for param, teacher_value in teacher_values:
            backups.append((param, param.detach().clone()))
            param.copy_(teacher_value.to(device=param.device, dtype=param.dtype))
        try:
            actor = getattr(worker, "actor", None)
            infer_batch = getattr(actor, "infer_batch", None)
            if callable(infer_batch):
                teacher_output = infer_batch(data)
            elif compute_fn is not None:
                teacher_output = compute_fn(worker, data)
            else:
                return None
        finally:
            for param, backup in backups:
                param.copy_(backup)
    tensor = _teacher_tensor_from_output(teacher_output)
    if tensor is None:
        return None
    try:
        return tensor.cpu()
    except Exception:
        return tensor


def _patch_actor_worker_internal_teacher() -> None:
    module_names = (
        "verl.workers.engine_workers",
        "verl.workers.fsdp_workers",
        "verl.workers.megatron_workers",
        "verl.workers.roles.actor_rollout_ref",
    )
    class_names = ("ActorRolloutRefWorker", "ActorWorker", "FSDPWorker")
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for class_name in class_names:
            worker_cls = getattr(module, class_name, None)
            if worker_cls is None or getattr(worker_cls, "_rlsd_internal_teacher_patched", False):
                continue
            original_compute = getattr(worker_cls, "compute_log_prob", None)
            if original_compute is None:
                continue

            @functools.wraps(original_compute)
            def compute_log_prob(self, data, _original_compute=original_compute):  # type: ignore[no-untyped-def]
                output = _original_compute(self, data)
                if _env_truthy("RLSD_INTERNAL_TEACHER_LOGPROB", False):
                    teacher_logprobs = _compute_internal_teacher_logprobs(self, data, _original_compute)
                    if teacher_logprobs is None:
                        message = "Internal EMA teacher logprobs were requested but could not be computed."
                        if _env_truthy("RLSD_INTERNAL_TEACHER_STRICT", True):
                            raise RuntimeError(message)
                        logger.warning(message)
                    else:
                        _set_item(output, "teacher_logprobs", teacher_logprobs)
                return output

            worker_cls.compute_log_prob = compute_log_prob

            original_update = getattr(worker_cls, "update_actor", None)
            if original_update is not None:

                @functools.wraps(original_update)
                def update_actor(self, data, _original_update=original_update):  # type: ignore[no-untyped-def]
                    result = _original_update(self, data)
                    try:
                        _update_internal_teacher_ema(self)
                    except Exception as exc:
                        if _env_truthy("RLSD_INTERNAL_TEACHER_STRICT", True):
                            raise
                        logger.warning("Internal EMA teacher update failed: %s", exc)
                    return result

                worker_cls.update_actor = update_actor

            worker_cls._rlsd_internal_teacher_patched = True
            logger.info("Patched %s.%s for internal EMA teacher logprobs.", module_name, class_name)


def _patch_checkpoint_engine_update_weights() -> None:
    try:
        checkpoint_mod = importlib.import_module("verl.checkpoint_engine.base")
    except Exception:
        return
    manager_cls = getattr(checkpoint_mod, "CheckpointEngineManager", None)
    if manager_cls is None or getattr(manager_cls, "_rlsd_teacher_ema_patched", False):
        return
    original = manager_cls.update_weights

    async def patched_update_weights(self, global_steps: int | None = None):  # type: ignore[no-untyped-def]
        result = await original(self, global_steps=global_steps)
        try:
            _maybe_sync_from_registered_trainer()
        except Exception as exc:
            logger.warning("Teacher EMA sync failed after rollout weight update: %s", exc)
        return result

    manager_cls.update_weights = patched_update_weights
    manager_cls._rlsd_teacher_ema_patched = True


def _patch_ray_trainer_init() -> None:
    try:
        ray_trainer_mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
    except Exception:
        return
    trainer_cls = getattr(ray_trainer_mod, "RayPPOTrainer", None)
    if trainer_cls is None or getattr(trainer_cls, "_rlsd_teacher_ema_patched", False):
        return
    original_init = trainer_cls.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        controller = TeacherEMAController.from_trainer(self)
        register_teacher_ema_controller(controller, self)

    trainer_cls.__init__ = patched_init
    trainer_cls._rlsd_teacher_ema_patched = True


def _unwrap_ray_actor_class(cls: Any) -> Any:
    """Return the underlying Python class for a @ray.remote actor wrapper."""
    inner = cls
    for _ in range(4):
        if hasattr(inner, "__ray_actor_class__"):
            inner = inner.__ray_actor_class__
            if hasattr(inner, "__ray_metadata__"):
                meta = inner.__ray_metadata__
                modified = getattr(meta, "modified_class", None)
                if modified is not None:
                    inner = modified
            continue
        if hasattr(inner, "__wrapped__"):
            inner = inner.__wrapped__
            continue
        break
    return inner


def _patch_teacher_model_manager_lora() -> None:
    try:
        teacher_model_mod = importlib.import_module("verl.experimental.teacher_loop.teacher_model")
    except Exception:
        return
    manager_cls = getattr(teacher_model_mod, "TeacherModelManager", None)
    if manager_cls is None or getattr(manager_cls, "_rlsd_teacher_ema_patched", False):
        return
    original_init = manager_cls._initialize_llm_servers

    def patched_initialize(self):  # type: ignore[no-untyped-def]
        if os.environ.get("RLSD_TEACHER_EMA_ENABLED", "").lower() in {"1", "true", "yes"}:
            lora_rank = int(os.environ.get("RLSD_TEACHER_EMA_LORA_RANK", "0") or 0)
            if lora_rank > 0 and not getattr(teacher_model_mod, "_rlsd_hf_model_config_patched", False):
                original_hf = teacher_model_mod.HFModelConfig

                class TeacherEMAHFModelConfig(original_hf):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        rank = int(os.environ.get("RLSD_TEACHER_EMA_LORA_RANK", "0") or 0)
                        if rank > 0:
                            self.lora_rank = rank

                teacher_model_mod.HFModelConfig = TeacherEMAHFModelConfig
                teacher_model_mod._rlsd_hf_model_config_patched = True
        return original_init(self)

    manager_cls._initialize_llm_servers = patched_initialize
    manager_cls._rlsd_teacher_ema_patched = True


def _patch_teacher_vllm_server_lora() -> None:
    try:
        server_mod = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    except Exception:
        return
    server_cls = getattr(server_mod, "AsyncvLLMServer", None)
    if server_cls is None:
        return

    target_cls = _unwrap_ray_actor_class(server_cls)
    if getattr(target_cls, "_rlsd_teacher_ema_patched", False):
        return

    original_property = getattr(target_cls, "lora_as_adapter", None)
    if original_property is None:
        logger.info(
            "Skip teacher EMA vLLM lora_as_adapter patch: %s has no lora_as_adapter "
            "(installed veRL may not expose this hook; EMA HTTP adapter load still attempted).",
            getattr(target_cls, "__name__", repr(target_cls)),
        )
        return
    if not isinstance(original_property, property):
        logger.warning(
            "Skip teacher EMA vLLM lora_as_adapter patch: attribute is %s, not property.",
            type(original_property).__name__,
        )
        return

    def lora_as_adapter(self):  # type: ignore[no-untyped-def]
        if bool(getattr(self, "is_teacher_model", False)) and os.environ.get(
            "RLSD_TEACHER_EMA_ENABLED", ""
        ).lower() in {"1", "true", "yes"}:
            return True
        return original_property.fget(self)

    target_cls.lora_as_adapter = property(lora_as_adapter)
    target_cls._rlsd_teacher_ema_patched = True
    logger.info("Patched %s.lora_as_adapter for teacher EMA LoRA.", getattr(target_cls, "__name__", target_cls))


def patch_teacher_ema() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_actor_worker_export()
    _patch_actor_worker_internal_teacher()
    _patch_teacher_model_manager_lora()
    _patch_ray_trainer_init()
    _patch_checkpoint_engine_update_weights()
    _patch_teacher_vllm_server_lora()
    _PATCHED = True
