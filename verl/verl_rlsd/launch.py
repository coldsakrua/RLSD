from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from . import advantage as custom_advantage
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


def main() -> None:
    _patch_compute_advantage()
    patch_teacher_ema()
    module_name = os.environ.get("VERL_MAIN_MODULE", "verl.trainer.main_ppo")
    module = importlib.import_module(module_name)
    entry = getattr(module, "main", None)
    if entry is None:
        raise RuntimeError(f"{module_name} does not expose a main() function.")
    sys.exit(entry())


if __name__ == "__main__":
    main()
