import re
from typing import Any, Dict, List, Sequence

import torch

from reward_fn import verifiable_math_reward
from rlsd_trainer import RLSDTrainer


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_TRAILING_SIMPLE_ANSWER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:answer|final answer)?\s*:?\s*([A-E]|[-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*$"
)


class RLSDSignFallbackTrainer(RLSDTrainer):
    """
    Sign-constrained fallback RLSD:
    - mixed group: reward sign controls direction, teacher only scales magnitude
    - all-correct group: positive-only fallback
    - all-wrong group: negative-only fallback
    """

    def __init__(
        self,
        *args,
        lambda_plus: float = 0.05,
        lambda_minus: float = 0.05,
        lambda_plus_min: float = 0.0,
        lambda_minus_min: float = 0.0,
        fallback_decay_steps: int = 200,
        fallback_eps0: float = 0.05,
        adv_clip_low: float = -1.0,
        adv_clip_high: float = 1.0,
        answer_token_downweight: float = 0.2,
        suppress_gt_shortcut: bool = True,
        reward_binary_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_plus = float(lambda_plus)
        self.lambda_minus = float(lambda_minus)
        self.lambda_plus_min = float(lambda_plus_min)
        self.lambda_minus_min = float(lambda_minus_min)
        self.fallback_decay_steps = int(fallback_decay_steps)
        self.fallback_eps0 = float(fallback_eps0)
        self.adv_clip_low = float(adv_clip_low)
        self.adv_clip_high = float(adv_clip_high)
        self.answer_token_downweight = float(answer_token_downweight)
        self.suppress_gt_shortcut = bool(suppress_gt_shortcut)
        self.reward_binary_threshold = float(reward_binary_threshold)

    def _current_fallback_lambda(self, start: float, min_value: float) -> float:
        start = float(start)
        min_value = float(min_value)
        if self.fallback_decay_steps <= 0:
            return start
        step = getattr(self.state, "global_step", 0)
        progress = min(max(step, 0), self.fallback_decay_steps) / float(self.fallback_decay_steps)
        return start + (min_value - start) * progress

    def _decode_completion_texts(
        self, completion_ids: torch.Tensor, completion_mask: torch.Tensor
    ) -> List[str]:
        tokenizer = self._get_tokenizer()
        texts: List[str] = []
        for ids_row, mask_row in zip(completion_ids, completion_mask):
            valid_ids = ids_row[mask_row.bool()].tolist()
            text = tokenizer.decode(valid_ids, skip_special_tokens=True)
            texts.append(text)
        return texts

    def _expand_column_for_completions(self, values: List[Any], target_len: int) -> List[Any]:
        if not values:
            return [""] * target_len
        if len(values) == target_len:
            return values
        if target_len % len(values) == 0:
            repeat = target_len // len(values)
            return [v for v in values for _ in range(repeat)]
        return [values[i % len(values)] for i in range(target_len)]

    def _compute_binary_rewards(
        self,
        inputs,
        completion_texts: List[str],
        target_len: int,
        completion_ids=None,
        completion_mask=None,
    ) -> torch.Tensor:
        device = self.accelerator.device
        raw_solutions = [x.get("solution", "") for x in inputs]
        solutions = self._expand_column_for_completions(raw_solutions, target_len)
        solution_texts = [s if isinstance(s, str) else str(s) for s in solutions]
        # Keep "count as correct" logic based on pure correctness (no format penalties).
        rewards = verifiable_math_reward(completion_texts, solution_texts)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        if reward_tensor.numel() != target_len:
            # Safety fallback: when mismatch happens, force neutral reward (all-wrong semantics).
            reward_tensor = torch.zeros(target_len, dtype=torch.float32, device=device)
        binary = (reward_tensor > self.reward_binary_threshold).float()
        return binary

    def _masked_minmax_normalize(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.bool()
        masked_min = torch.where(mask_bool, values, torch.full_like(values, float("inf"))).min(dim=1).values
        masked_max = torch.where(mask_bool, values, torch.full_like(values, float("-inf"))).max(dim=1).values
        no_valid = mask_bool.sum(dim=1) == 0
        masked_min = torch.where(no_valid, torch.zeros_like(masked_min), masked_min)
        masked_max = torch.where(no_valid, torch.ones_like(masked_max), masked_max)
        denom = (masked_max - masked_min).clamp(min=1e-6).unsqueeze(1)
        normalized = (values - masked_min.unsqueeze(1)) / denom
        return normalized * mask

    def _length_normalize_mean_abs(self, adv: torch.Tensor, mask: torch.Tensor, target_mean: float) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1.0)
        mean_abs = (adv.abs() * mask).sum(dim=1) / lengths
        scale = (float(target_mean) / mean_abs.clamp(min=1e-6)).unsqueeze(1)
        return adv * scale * mask

    def _find_answer_spans(self, text: str) -> List[tuple[int, int]]:
        spans: List[tuple[int, int]] = []
        for m in _ANSWER_TAG_RE.finditer(text):
            spans.append((m.start(), m.end()))
        for m in _BOXED_RE.finditer(text):
            spans.append((m.start(), m.end()))
        trailing = _TRAILING_SIMPLE_ANSWER_RE.search(text)
        if trailing is not None:
            spans.append((trailing.start(1), trailing.end(1)))
        return spans

    def _build_answer_token_weight_mask(
        self,
        completion_texts: List[str],
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        n, max_len = completion_mask.shape
        device = completion_mask.device
        weights = torch.ones((n, max_len), dtype=torch.float32, device=device)

        if (not self.suppress_gt_shortcut) or self.answer_token_downweight >= 0.999:
            return weights

        tokenizer = self._get_tokenizer()
        use_offset = getattr(tokenizer, "is_fast", False)
        down = float(self.answer_token_downweight)

        for i, text in enumerate(completion_texts):
            spans = self._find_answer_spans(text)
            if not spans:
                continue
            valid_len = int(completion_mask[i].sum().item())
            if valid_len <= 0:
                continue

            if use_offset:
                try:
                    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                    offsets = enc.get("offset_mapping", [])
                    row_w = torch.ones(valid_len, dtype=torch.float32, device=device)
                    limit = min(valid_len, len(offsets))
                    for t in range(limit):
                        tok_s, tok_e = offsets[t]
                        if tok_e <= tok_s:
                            continue
                        for span_s, span_e in spans:
                            if tok_s < span_e and tok_e > span_s:
                                row_w[t] = down
                                break
                    weights[i, :valid_len] = row_w
                    continue
                except Exception:
                    pass

            # Fallback when offset mapping is unavailable: downweight tail tokens.
            tail_k = min(valid_len, 8)
            weights[i, valid_len - tail_k : valid_len] = down
        return weights

    def _generate_and_score_completions(self, inputs):
        batch = super()._generate_and_score_completions(inputs)

        seq_advantages = batch["advantages"]
        if seq_advantages.dim() != 1:
            return batch

        completion_mask = batch["completion_mask"].float()
        completion_ids = batch["completion_ids"]
        sample_count = seq_advantages.numel()
        if sample_count == 0 or sample_count % self.num_generations != 0:
            return batch

        student_logps = self._compute_student_logps(batch)
        teacher_prompts = self._build_teacher_prompts(inputs)
        teacher_prompts = self._expand_column_for_completions(teacher_prompts, sample_count)
        teacher_logps = self._compute_teacher_logps(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            teacher_prompts=teacher_prompts,
        )

        token_delta = (teacher_logps - student_logps).detach() * completion_mask
        clip_low = 1.0 - self.jsd_token_clip
        clip_high = 1.0 + self.jsd_token_clip

        completion_texts = self._decode_completion_texts(completion_ids, completion_mask)
        rewards_binary = self._compute_binary_rewards(
            inputs,
            completion_texts,
            sample_count,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )
        group_rewards = rewards_binary.view(-1, self.num_generations)
        all_correct_group = (group_rewards > 0.5).all(dim=1)
        all_wrong_group = (group_rewards < 0.5).all(dim=1)
        mixed_group = ~(all_correct_group | all_wrong_group)

        all_correct = all_correct_group.repeat_interleave(self.num_generations).unsqueeze(1)
        all_wrong = all_wrong_group.repeat_interleave(self.num_generations).unsqueeze(1)
        mixed = mixed_group.repeat_interleave(self.num_generations).unsqueeze(1)

        base_adv = seq_advantages.unsqueeze(1)
        sign = torch.sign(seq_advantages).unsqueeze(1)
        w_mixed = torch.exp(sign * token_delta)
        w_mixed = torch.clamp(w_mixed, min=clip_low, max=clip_high)
        alpha_mixed = self._current_lambda()
        mixed_adv = base_adv * ((1.0 - alpha_mixed) + alpha_mixed * w_mixed)
        mixed_rollout_mask = self._rollout_mask(seq_advantages).unsqueeze(1)
        mixed_adv = torch.where(mixed_rollout_mask, mixed_adv, base_adv)

        w_plus = torch.clamp(torch.exp(token_delta), min=clip_low, max=clip_high)
        plus_norm = self._masked_minmax_normalize(w_plus, completion_mask)
        plus_raw = (self.fallback_eps0 + plus_norm) * completion_mask
        lambda_plus_now = self._current_fallback_lambda(self.lambda_plus, self.lambda_plus_min)
        plus_adv = self._length_normalize_mean_abs(plus_raw, completion_mask, lambda_plus_now)

        support = self._masked_minmax_normalize(torch.exp(torch.clamp(token_delta, min=-20.0, max=20.0)), completion_mask)
        minus_raw = -(self.fallback_eps0 + (1.0 - support)) * completion_mask
        lambda_minus_now = self._current_fallback_lambda(self.lambda_minus, self.lambda_minus_min)
        minus_adv = self._length_normalize_mean_abs(minus_raw, completion_mask, lambda_minus_now)

        token_advantages = torch.zeros_like(mixed_adv)
        token_advantages = torch.where(mixed, mixed_adv, token_advantages)
        token_advantages = torch.where(all_correct, plus_adv, token_advantages)
        token_advantages = torch.where(all_wrong, minus_adv, token_advantages)

        answer_weight_mask = self._build_answer_token_weight_mask(completion_texts, completion_mask)
        token_advantages = token_advantages * answer_weight_mask
        token_advantages = torch.clamp(token_advantages, min=self.adv_clip_low, max=self.adv_clip_high)
        token_advantages = token_advantages * completion_mask

        batch["advantages"] = token_advantages

        self._log_metric("rlsd/mixed_alpha", alpha_mixed)
        self._log_metric("rlsd/lambda_plus", lambda_plus_now)
        self._log_metric("rlsd/lambda_minus", lambda_minus_now)
        self._log_metric("rlsd/group_all_correct_frac", float(all_correct_group.float().mean().item()))
        self._log_metric("rlsd/group_all_wrong_frac", float(all_wrong_group.float().mean().item()))
        self._log_metric("rlsd/group_mixed_frac", float(mixed_group.float().mean().item()))
        self._log_metric("rlsd/answer_weight_mean", float(answer_weight_mask.mean().item()))
        self._stash_rollout_for_checkpoint(
            inputs,
            completion_ids,
            completion_mask,
            reward_values=rewards_binary.detach().cpu().tolist(),
            seq_advantages_1d=seq_advantages,
            token_advantages=token_advantages,
        )
        return batch
