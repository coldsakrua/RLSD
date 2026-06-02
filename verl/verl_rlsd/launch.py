from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any

from . import advantage as custom_advantage
from .rollout_metrics import patch_rollout_length_logging
from .teacher_ema import patch_teacher_ema


def _cfg_get(config: Any, path: str, default: Any = None) -> Any:
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _patch_compute_advantage() -> None:
    try:
        ray_trainer = importlib.import_module("verl.trainer.ppo.ray_trainer")
    except Exception:
        return
    original = getattr(ray_trainer, "compute_advantage", None)
    if original is None:
        return

    def patched_compute_advantage(data: Any, adv_estimator: Any, gamma: float = 1.0, lam: float = 1.0, num_repeat: int = 1, config: Any = None, **kwargs: Any):
        name = getattr(adv_estimator, "value", adv_estimator)
        custom_name = _cfg_get(config, "algorithm.rlsd.custom_adv_estimator", None)
        if custom_name:
            name = str(custom_name)
        if str(name) in custom_advantage.CUSTOM_ADV_ESTIMATORS:
            return custom_advantage.compute_custom_advantage(data, str(name), config)
        return original(
            data=data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            config=config,
            **kwargs,
        )

    ray_trainer.compute_advantage = patched_compute_advantage


_RAY_INIT_OLD = "ray.init(namespace=namespace)"
_RAY_INIT_NEW = (
    'ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), '
    "namespace=namespace, ignore_reinit_error=True)"
)


def _patch_verl_vllm_async_server_on_disk() -> None:
    """Patch installed verl so vLLM EngineCore subprocess joins the training Ray cluster."""
    try:
        mod = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    except Exception:
        return
    path = getattr(mod, "__file__", None)
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RAY_INIT_NEW in text:
        return
    if _RAY_INIT_OLD not in text:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(_RAY_INIT_OLD, _RAY_INIT_NEW, 1))


_FSDP_VLLM_ADD_LORA_OLD = "self.inference_engine.llm_engine.add_lora(lora_reqest)"
_FSDP_VLLM_ADD_LORA_CALL = (
    '__import__("verl_rlsd.launch", fromlist=["_rlsd_add_lora"]).'
    '_rlsd_add_lora(self, lora_reqest)  # RLSD_ASYNC_VLLM_ADD_LORA_PATCH'
)


def _rlsd_add_lora(sharding_manager: Any, lora_request: Any) -> None:
    """Compatibility wrapper for veRL LoRA sync across vLLM v0/v1 objects."""
    import asyncio
    import inspect

    inference_engine = getattr(sharding_manager, "inference_engine", None)
    candidates = [
        getattr(inference_engine, "llm_engine", None),
        inference_engine,
        getattr(inference_engine, "worker", None),
        getattr(getattr(inference_engine, "worker", None), "model_runner", None),
        getattr(sharding_manager, "model_runner", None),
    ]
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        add_lora = getattr(candidate, "add_lora", None)
        if add_lora is None:
            continue
        result = add_lora(lora_request)
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is None:
                asyncio.run(result)
            elif loop.is_running():
                asyncio.run_coroutine_threadsafe(result, loop).result()
            else:
                loop.run_until_complete(result)
        return
    raise AttributeError(
        "RLSD failed to find a vLLM add_lora API; this veRL/vLLM combination "
        "does not expose llm_engine.add_lora or an equivalent add_lora method."
    )


def _find_module_file(module_name: str, relative_path: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
        path = getattr(spec, "origin", None) if spec is not None else None
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    for base in sys.path:
        if not base:
            continue
        path = os.path.join(base, relative_path)
        if os.path.isfile(path):
            return path
    return None


def _repair_bad_add_lora_patch(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "RLSD_ASYNC_VLLM_ADD_LORA_PATCH" in line and "_rlsd_add_lora" not in line:
            start = i
            end = None
            limit = min(len(lines), i + 80)
            j = i + 1
            while j < limit:
                if "_rlsd_loop.run_until_complete(_rlsd_lora_result)" in lines[j]:
                    end = j + 1
                    break
                j += 1
            if end is None:
                out.append(line)
                i = start + 1
                continue
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}{_FSDP_VLLM_ADD_LORA_CALL}{newline}")
            changed = True
            i = end
            continue
        out.append(line)
        i += 1
    return "".join(out), changed


def _patch_fsdp_vllm_add_lora_on_disk() -> None:
    """Patch old veRL FSDP-vLLM sharding to support vLLM v1 LoRA sync."""
    path = _find_module_file(
        "verl.workers.sharding_manager.fsdp_vllm",
        os.path.join("verl", "workers", "sharding_manager", "fsdp_vllm.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text, repaired = _repair_bad_add_lora_patch(text)
    if not repaired and "RLSD_ASYNC_VLLM_ADD_LORA_PATCH" in text:
        return
    if not repaired and _FSDP_VLLM_ADD_LORA_OLD not in text:
        return
    if not repaired:
        text = text.replace(_FSDP_VLLM_ADD_LORA_OLD, _FSDP_VLLM_ADD_LORA_CALL, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_ray_init_for_vllm() -> None:
    """Ensure Ray workers inherit the vLLM mode and training cluster RAY_ADDRESS."""
    os.environ.setdefault("VLLM_USE_V1", "1")
    try:
        import ray
    except Exception:
        return
    if getattr(ray.init, "_rlsd_vllm_env_patched", False):
        return
    original_init = ray.init

    def patched_init(*args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars.setdefault("VLLM_USE_V1", os.environ.get("VLLM_USE_V1", "1"))
        if os.environ.get("RLSD_MERGE_LORA_FOR_ASYNC_VLLM"):
            env_vars.setdefault(
                "RLSD_MERGE_LORA_FOR_ASYNC_VLLM",
                os.environ["RLSD_MERGE_LORA_FOR_ASYNC_VLLM"],
            )
        if os.environ.get("RAY_ADDRESS"):
            env_vars.setdefault("RAY_ADDRESS", os.environ["RAY_ADDRESS"])
        if os.environ.get("RAY_TMPDIR"):
            env_vars.setdefault("RAY_TMPDIR", os.environ["RAY_TMPDIR"])
        runtime_env["env_vars"] = env_vars
        kwargs["runtime_env"] = runtime_env
        result = original_init(*args, **kwargs)
        try:
            gcs = ray.get_runtime_context().gcs_address
            if gcs:
                os.environ["RAY_ADDRESS"] = gcs
        except Exception:
            pass
        return result

    patched_init._rlsd_vllm_env_patched = True
    ray.init = patched_init


def main() -> None:
    _patch_verl_vllm_async_server_on_disk()
    _patch_fsdp_vllm_add_lora_on_disk()
    _patch_ray_init_for_vllm()
    _patch_compute_advantage()
    patch_rollout_length_logging()
    patch_teacher_ema()
    module_name = os.environ.get("VERL_MAIN_MODULE", "verl.trainer.main_ppo")
    module = importlib.import_module(module_name)
    entry = getattr(module, "main", None)
    if entry is None:
        raise RuntimeError(f"{module_name} does not expose a main() function.")
    sys.exit(entry())


if __name__ == "__main__":
    main()
