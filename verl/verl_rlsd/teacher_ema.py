from __future__ import annotations

import importlib
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
    if server_cls is None or getattr(server_cls, "_rlsd_teacher_ema_patched", False):
        return

    original_property = server_cls.lora_as_adapter

    def lora_as_adapter(self):  # type: ignore[no-untyped-def]
        if bool(getattr(self, "is_teacher_model", False)) and os.environ.get(
            "RLSD_TEACHER_EMA_ENABLED", ""
        ).lower() in {"1", "true", "yes"}:
            return True
        return original_property.fget(self)

    server_cls.lora_as_adapter = property(lora_as_adapter)
    server_cls._rlsd_teacher_ema_patched = True


def patch_teacher_ema() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_actor_worker_export()
    _patch_teacher_model_manager_lora()
    _patch_ray_trainer_init()
    _patch_checkpoint_engine_update_weights()
    _patch_teacher_vllm_server_lora()
    _PATCHED = True
