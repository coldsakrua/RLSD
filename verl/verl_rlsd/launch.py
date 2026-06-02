from __future__ import annotations

import importlib
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
_FSDP_VLLM_ADD_LORA_NEW = '''# RLSD_ASYNC_VLLM_ADD_LORA_PATCH
                    import inspect as _rlsd_inspect

                    _rlsd_lora_target = getattr(self.inference_engine, "llm_engine", None)
                    if _rlsd_lora_target is None:
                        _rlsd_lora_target = self.inference_engine
                    if not hasattr(_rlsd_lora_target, "add_lora") and hasattr(self, "model_runner"):
                        _rlsd_model_runner = getattr(self, "model_runner", None)
                        if hasattr(_rlsd_model_runner, "add_lora"):
                            _rlsd_lora_target = _rlsd_model_runner
                    if not hasattr(_rlsd_lora_target, "add_lora"):
                        raise RuntimeError(
                            "RLSD failed to find a vLLM add_lora API on "
                            f"{type(_rlsd_lora_target).__name__}; this veRL/vLLM "
                            "combination does not expose llm_engine.add_lora."
                        )
                    _rlsd_lora_result = _rlsd_lora_target.add_lora(lora_reqest)
                    if _rlsd_inspect.isawaitable(_rlsd_lora_result):
                        import asyncio as _rlsd_asyncio

                        try:
                            _rlsd_loop = _rlsd_asyncio.get_event_loop()
                        except RuntimeError:
                            _rlsd_loop = None
                        if _rlsd_loop is None:
                            _rlsd_asyncio.run(_rlsd_lora_result)
                        elif _rlsd_loop.is_running():
                            _rlsd_asyncio.run_coroutine_threadsafe(
                                _rlsd_lora_result,
                                _rlsd_loop,
                            ).result()
                        else:
                            _rlsd_loop.run_until_complete(_rlsd_lora_result)'''


def _patch_fsdp_vllm_add_lora_on_disk() -> None:
    """Patch old veRL FSDP-vLLM sharding to support vLLM v1 LoRA sync."""
    try:
        mod = importlib.import_module("verl.workers.sharding_manager.fsdp_vllm")
    except Exception:
        return
    path = getattr(mod, "__file__", None)
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if "RLSD_ASYNC_VLLM_ADD_LORA_PATCH" in text:
        return
    if _FSDP_VLLM_ADD_LORA_OLD not in text:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(_FSDP_VLLM_ADD_LORA_OLD, _FSDP_VLLM_ADD_LORA_NEW, 1))


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
