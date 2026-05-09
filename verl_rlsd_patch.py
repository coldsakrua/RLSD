import os
from typing import Dict, List

import numpy as np
import torch


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "")
    if value == "":
        return float(default)
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if value == "":
        return int(default)
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "")
    return value if value != "" else default


def _to_name(estimator) -> str:
    if hasattr(estimator, "value"):
        return str(estimator.value)
    return str(estimator)


def _group_positions(index) -> List[List[int]]:
    arr = np.asarray(index, dtype=object)
    groups: Dict[str, List[int]] = {}
    for i, uid in enumerate(arr.tolist()):
        key = str(uid)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    return list(groups.values())


def _compute_grpo_seq_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    norm_adv_by_std_in_grpo: bool = True,
) -> torch.Tensor:
    scores = (token_level_rewards * response_mask).sum(dim=-1)
    advantages = torch.zeros_like(scores)
    groups = _group_positions(index)
    for pos_list in groups:
        idx = torch.tensor(pos_list, device=scores.device, dtype=torch.long)
        g_scores = scores[idx]
        mean = g_scores.mean()
        if norm_adv_by_std_in_grpo:
            std = g_scores.std(unbiased=False).clamp(min=1e-6)
            advantages[idx] = (g_scores - mean) / std
        else:
            advantages[idx] = g_scores - mean
    return advantages


def _current_lambda(start: float, decay_steps: int, step: int) -> float:
    if decay_steps <= 0:
        return float(start)
    p = min(max(step, 0), decay_steps) / float(decay_steps)
    return float(start) * (1.0 - p)


def _current_linear(start: float, end: float, decay_steps: int, step: int) -> float:
    if decay_steps <= 0:
        return float(start)
    p = min(max(step, 0), decay_steps) / float(decay_steps)
    return float(start) + (float(end) - float(start)) * p


def _rowwise_minmax_01(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask.bool()
    row_min = torch.where(mask_bool, values, torch.full_like(values, float("inf"))).min(dim=1).values
    row_max = torch.where(mask_bool, values, torch.full_like(values, float("-inf"))).max(dim=1).values
    no_valid = mask_bool.sum(dim=1) == 0
    row_min = torch.where(no_valid, torch.zeros_like(row_min), row_min)
    row_max = torch.where(no_valid, torch.ones_like(row_max), row_max)
    denom = (row_max - row_min).clamp(min=1e-6).unsqueeze(1)
    out = (values - row_min.unsqueeze(1)) / denom
    return out * mask


def _normalize_mean_abs(adv: torch.Tensor, mask: torch.Tensor, target_mean_abs: float) -> torch.Tensor:
    lengths = mask.sum(dim=1).clamp(min=1.0)
    mean_abs = (adv.abs() * mask).sum(dim=1) / lengths
    scale = (float(target_mean_abs) / mean_abs.clamp(min=1e-6)).unsqueeze(1)
    return adv * scale * mask


def _tail_downweight_mask(mask: torch.Tensor, tail_k: int, downweight: float) -> torch.Tensor:
    if tail_k <= 0 or downweight >= 0.999:
        return torch.ones_like(mask)
    n, _ = mask.shape
    w = torch.ones_like(mask)
    for i in range(n):
        valid_len = int(mask[i].sum().item())
        if valid_len <= 0:
            continue
        k = min(valid_len, tail_k)
        w[i, valid_len - k : valid_len] = downweight
    return w


def _get_binary_rewards(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    non_tensor_batch,
    threshold: float,
) -> torch.Tensor:
    if non_tensor_batch is not None and "is_correct" in non_tensor_batch:
        arr = np.asarray(non_tensor_batch["is_correct"]).astype(np.float32)
        return torch.tensor(arr, device=token_level_rewards.device)
    seq_reward = (token_level_rewards * response_mask).sum(dim=1)
    return (seq_reward > float(threshold)).float()


def _rollout_filter_mask(seq_adv: torch.Tensor, num_groups: int, num_generations: int, mode: str) -> torch.Tensor:
    if mode == "all":
        return torch.ones_like(seq_adv, dtype=torch.bool)
    if mode == "positive":
        return seq_adv > 0
    if mode == "negative":
        return seq_adv < 0
    if mode == "mixed":
        if seq_adv.numel() != num_groups * num_generations:
            return torch.ones_like(seq_adv, dtype=torch.bool)
        grouped = seq_adv.view(num_groups, num_generations)
        has_pos = (grouped > 0).any(dim=1)
        has_neg = (grouped < 0).any(dim=1)
        return (has_pos & has_neg).repeat_interleave(num_generations)
    return torch.ones_like(seq_adv, dtype=torch.bool)


def _is_custom_estimator(name: str) -> bool:
    return name in {"rlsd_verl", "rlsd_strict_verl"}


def _silence_math_verify_logger() -> None:
    """math_verify 内部 grader 在某些 SymPy 输入下会抛 AttributeError 等异常，
    它会自己捕获并把异常以 logger.exception 打到日志里，对训练流程无害。
    但这种 stack trace 会刷屏并污染 stdout，这里把它的 logger 噪声压住。"""
    import logging

    for name in ("math_verify", "math_verify.grader", "math_verify.utils"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def _import_advantage_estimator():
    """新版 verl 把 AdvantageEstimator 放到了 ppo.core_algos，旧版可能在 trainer.config。
    这里做个兼容 import。"""
    try:
        from verl.trainer.ppo.core_algos import AdvantageEstimator  # type: ignore

        return AdvantageEstimator
    except Exception:
        pass
    try:
        from verl.trainer.config import AdvantageEstimator  # type: ignore

        return AdvantageEstimator
    except Exception:
        return None


def _set_estimator(config, value) -> None:
    """OmegaConf 默认是 struct 模式，需要先 open_dict 才能改 algorithm.adv_estimator。"""
    try:
        from omegaconf import OmegaConf, open_dict

        if OmegaConf.is_config(config):
            with open_dict(config):
                config.algorithm.adv_estimator = value
            return
    except Exception:
        pass
    config.algorithm.adv_estimator = value


def patch_verl_compute_advantage():
    _silence_math_verify_logger()

    AdvantageEstimator = _import_advantage_estimator()
    from verl.trainer.ppo import ray_trainer as ray_trainer_mod

    original_compute_advantage = ray_trainer_mod.compute_advantage
    grpo_value = (
        AdvantageEstimator.GRPO.value
        if AdvantageEstimator is not None and hasattr(AdvantageEstimator, "GRPO")
        else "grpo"
    )

    original_init = ray_trainer_mod.RayPPOTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        original_estimator = None
        swapped = False
        if config is not None:
            try:
                original_estimator = config.algorithm.adv_estimator
            except Exception:
                original_estimator = None
            name = _to_name(original_estimator) if original_estimator is not None else ""
            if _is_custom_estimator(name):
                _set_estimator(config, grpo_value)
                swapped = True
        try:
            original_init(self, *args, **kwargs)
        finally:
            if swapped:
                _set_estimator(config, original_estimator)

    ray_trainer_mod.RayPPOTrainer.__init__ = _patched_init

    def _patched_compute_advantage(
        data,
        adv_estimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config=None,
        **extra_kwargs,
    ):
        name = _to_name(adv_estimator)
        if not _is_custom_estimator(name):
            return original_compute_advantage(
                data=data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
                **extra_kwargs,
            )

        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = ray_trainer_mod.compute_response_mask(data)

        response_mask = data.batch["response_mask"].float()
        token_level_rewards = data.batch["token_level_rewards"].float()
        uid = data.non_tensor_batch["uid"] if "uid" in data.non_tensor_batch else np.arange(len(response_mask))

        seq_adv = _compute_grpo_seq_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=uid,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        base_adv = seq_adv.unsqueeze(1)

        old_log_probs = data.batch.get("old_log_probs")
        ref_log_prob = data.batch.get("ref_log_prob")
        if old_log_probs is None:
            old_log_probs = torch.zeros_like(token_level_rewards)
        if ref_log_prob is None:
            ref_log_prob = old_log_probs
        g = (ref_log_prob - old_log_probs).detach() * response_mask

        jsd_clip = _env_float("RLSD_JSD_TOKEN_CLIP", 0.05)
        clip_low = 1.0 - jsd_clip
        clip_high = 1.0 + jsd_clip
        step = 0
        if config is not None and "global_steps" in data.meta_info:
            step = int(data.meta_info.get("global_steps", 0))

        binary_threshold = _env_float("RLSD_BINARY_THRESHOLD", 0.5)
        reward_binary = _get_binary_rewards(token_level_rewards, response_mask, data.non_tensor_batch, binary_threshold)
        reward_sign = (2.0 * reward_binary - 1.0).unsqueeze(1)
        adv_sign = torch.sign(base_adv)
        sign = torch.where(adv_sign == 0, reward_sign, adv_sign)

        mixed_lambda = _env_float("RLSD_MIXED_LAMBDA", 0.5)
        mixed_decay = _env_int("RLSD_MIXED_DECAY_STEPS", 50)
        alpha_mixed = _current_lambda(mixed_lambda, mixed_decay, step)
        w_mixed = torch.clamp(torch.exp(sign * g), min=clip_low, max=clip_high)
        mixed_adv = base_adv * ((1.0 - alpha_mixed) + alpha_mixed * w_mixed)

        groups = _group_positions(uid)
        num_groups = len(groups)
        num_generations = max(1, len(response_mask) // max(1, num_groups))
        rollout_filter = _env_str("RLSD_ROLLOUT_FILTER", "all")
        mixed_keep = _rollout_filter_mask(seq_adv, num_groups, num_generations, rollout_filter).unsqueeze(1)
        mixed_adv = torch.where(mixed_keep, mixed_adv, base_adv)

        if name == "rlsd_verl":
            token_adv = mixed_adv * response_mask
            adv_low = _env_float("RLSD_ADV_CLIP_LOW", -10.0)
            adv_high = _env_float("RLSD_ADV_CLIP_HIGH", 10.0)
            token_adv = torch.clamp(token_adv, min=adv_low, max=adv_high)
            data.batch["advantages"] = token_adv
            data.batch["returns"] = token_adv
            return data

        sample_all_correct = torch.zeros_like(reward_binary, dtype=torch.bool)
        sample_all_wrong = torch.zeros_like(reward_binary, dtype=torch.bool)
        sample_mixed = torch.zeros_like(reward_binary, dtype=torch.bool)
        for pos_list in groups:
            idx = torch.tensor(pos_list, device=reward_binary.device, dtype=torch.long)
            group_bin = reward_binary[idx]
            if bool((group_bin > 0.5).all().item()):
                sample_all_correct[idx] = True
            elif bool((group_bin < 0.5).all().item()):
                sample_all_wrong[idx] = True
            else:
                sample_mixed[idx] = True

        all_correct = sample_all_correct.unsqueeze(1)
        all_wrong = sample_all_wrong.unsqueeze(1)
        mixed = sample_mixed.unsqueeze(1)

        lambda_plus = _env_float("RLSD_LAMBDA_PLUS", 0.03)
        lambda_minus = _env_float("RLSD_LAMBDA_MINUS", 0.03)
        lambda_plus_min = _env_float("RLSD_LAMBDA_PLUS_MIN", 0.0)
        lambda_minus_min = _env_float("RLSD_LAMBDA_MINUS_MIN", 0.0)
        fallback_decay = _env_int("RLSD_FALLBACK_DECAY_STEPS", 200)
        fallback_eps0 = _env_float("RLSD_FALLBACK_EPS0", 0.05)

        lambda_plus_now = _current_linear(lambda_plus, lambda_plus_min, fallback_decay, step)
        lambda_minus_now = _current_linear(lambda_minus, lambda_minus_min, fallback_decay, step)

        w_plus = torch.clamp(torch.exp(g), min=clip_low, max=clip_high)
        plus_01 = _rowwise_minmax_01(w_plus, response_mask)
        plus_raw = (fallback_eps0 + plus_01) * response_mask
        plus_adv = _normalize_mean_abs(plus_raw, response_mask, lambda_plus_now)

        support = _rowwise_minmax_01(torch.exp(torch.clamp(g, min=-20.0, max=20.0)), response_mask)
        minus_raw = -(fallback_eps0 + (1.0 - support)) * response_mask
        minus_adv = _normalize_mean_abs(minus_raw, response_mask, lambda_minus_now)

        token_adv = torch.zeros_like(mixed_adv)
        token_adv = torch.where(mixed, mixed_adv, token_adv)
        token_adv = torch.where(all_correct, plus_adv, token_adv)
        token_adv = torch.where(all_wrong, minus_adv, token_adv)

        suppress_shortcut = _env_str("RLSD_SUPPRESS_GT_SHORTCUT", "true").lower() in {"1", "true", "yes", "y"}
        if suppress_shortcut:
            tail_k = _env_int("RLSD_TAIL_DOWNWEIGHT_TOKENS", 8)
            down = _env_float("RLSD_ANSWER_TOKEN_DOWNWEIGHT", 0.2)
            token_adv = token_adv * _tail_downweight_mask(response_mask, tail_k=tail_k, downweight=down)

        adv_low = _env_float("RLSD_ADV_CLIP_LOW", -1.0)
        adv_high = _env_float("RLSD_ADV_CLIP_HIGH", 1.0)
        token_adv = torch.clamp(token_adv, min=adv_low, max=adv_high) * response_mask
        data.batch["advantages"] = token_adv
        data.batch["returns"] = token_adv
        return data

    ray_trainer_mod.compute_advantage = _patched_compute_advantage
