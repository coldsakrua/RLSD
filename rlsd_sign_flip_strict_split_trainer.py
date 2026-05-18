import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from trl import GRPOTrainer

from reward_fn import verifiable_math_reward
from rlsd_trainer import RLSDTrainer


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_TAIL_ANSWER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:answer|final answer)?\s*:?\s*([A-E]|[-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*$"
)


class RLSDSignFlipStrictSplitTrainer(RLSDTrainer):
    """
    Strict split OPSD sign-flip variant:
    - mixed groups use the original GRPO sequence advantage as base A
    - all-correct/all-wrong groups use explicit signed fallback base advantages
    - token shaping uses the current clipped-weight parameters
    - positive-base tokens with g < 0 are flipped to negative advantage
    - with strict_split_mixed_only=True, all-correct/all-wrong groups are logged but receive zero feedback
    """

    def __init__(
        self,
        *args,
        all_correct_base_advantage: float = 1.0,
        all_wrong_base_advantage: float = -1.0,
        correct_weight_clip_low: float = 0.8,
        correct_weight_clip_high: float = 1.05,
        wrong_weight_clip_low: float = 0.95,
        wrong_weight_clip_high: float = 1.2,
        adv_clip_low: float = -1.2,
        adv_clip_high: float = 1.2,
        answer_token_downweight: float = 1.0,
        suppress_gt_shortcut: bool = True,
        reward_binary_threshold: float = 0.5,
        fallback_tail_tokens: int = 8,
        strict_split_mixed_only: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.all_correct_base_advantage = float(all_correct_base_advantage)
        self.all_wrong_base_advantage = float(all_wrong_base_advantage)
        self.correct_weight_clip_low = float(correct_weight_clip_low)
        self.correct_weight_clip_high = float(correct_weight_clip_high)
        self.wrong_weight_clip_low = float(wrong_weight_clip_low)
        self.wrong_weight_clip_high = float(wrong_weight_clip_high)
        self.adv_clip_low = float(adv_clip_low)
        self.adv_clip_high = float(adv_clip_high)
        self.answer_token_downweight = float(answer_token_downweight)
        self.suppress_gt_shortcut = bool(suppress_gt_shortcut)
        self.reward_binary_threshold = float(reward_binary_threshold)
        self.fallback_tail_tokens = int(fallback_tail_tokens)
        self.strict_split_mixed_only = bool(strict_split_mixed_only)

    def _expand_to_samples(self, values: Sequence[Any], target_len: int) -> List[Any]:
        if not values:
            return [""] * target_len
        values = list(values)
        if len(values) == target_len:
            return values
        if target_len % len(values) == 0:
            r = target_len // len(values)
            return [v for v in values for _ in range(r)]
        return [values[i % len(values)] for i in range(target_len)]

    def _compute_binary_rewards(
        self,
        inputs,
        completions: List[str],
        sample_count: int,
        completion_ids: Optional[torch.Tensor] = None,
        completion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = self.accelerator.device
        solutions = self._expand_to_samples([x.get("solution", "") for x in inputs], sample_count)
        solutions = [s if isinstance(s, str) else str(s) for s in solutions]
        # Keep "count as correct" logic based on pure correctness (no format penalties).
        rewards = verifiable_math_reward(completions, solutions)

        reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        if reward_t.numel() != sample_count:
            reward_t = torch.zeros(sample_count, dtype=torch.float32, device=device)
        return (reward_t > self.reward_binary_threshold).float()

    def _answer_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for m in _ANSWER_TAG_RE.finditer(text):
            spans.append((m.start(), m.end()))
        for m in _BOXED_RE.finditer(text):
            spans.append((m.start(), m.end()))
        tail = _TAIL_ANSWER_RE.search(text)
        if tail is not None:
            spans.append((tail.start(1), tail.end(1)))
        return spans

    def _answer_weight_mask(
        self,
        completion_texts: List[str],
        completion_mask: torch.Tensor,
        *,
        decode_length_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n, max_len = completion_mask.shape
        device = completion_mask.device
        weights = torch.ones((n, max_len), dtype=torch.float32, device=device)

        if not self.suppress_gt_shortcut or self.answer_token_downweight >= 0.999:
            return weights

        down = float(self.answer_token_downweight)
        tokenizer = self._get_tokenizer()
        use_offset = getattr(tokenizer, "is_fast", False)

        for i, text in enumerate(completion_texts):
            len_src = decode_length_mask if decode_length_mask is not None else completion_mask
            valid_len = int(len_src[i].sum().item())
            if valid_len <= 0:
                continue

            spans = self._answer_spans(text)
            if not spans:
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

            tail_k = min(valid_len, max(1, self.fallback_tail_tokens))
            weights[i, valid_len - tail_k : valid_len] = down
        return weights

    def _compute_teacher_logps_strict(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        teacher_prompts: Sequence[str],
    ) -> torch.Tensor:
        return super()._compute_teacher_logps(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            teacher_prompts=teacher_prompts,
        )

    def _generate_and_score_completions(self, inputs):
        # Call the parent class of RLSDTrainer directly to avoid reusing its weighting logic.
        batch = GRPOTrainer._generate_and_score_completions(self, inputs)

        seq_advantages = batch["advantages"]
        if seq_advantages.dim() != 1:
            return batch

        completion_mask = batch["completion_mask"].float()
        completion_ids = batch["completion_ids"]
        sample_count = seq_advantages.numel()
        if sample_count == 0 or sample_count % self.num_generations != 0:
            return batch

        # teacher-student token log-prob gap g_{i,t}
        student_logps = self._compute_student_logps(batch)
        teacher_prompts = self._expand_to_samples(self._build_teacher_prompts(inputs), sample_count)
        teacher_logps = self._compute_teacher_logps_strict(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            teacher_prompts=teacher_prompts,
        )
        g = (teacher_logps - student_logps).detach() * completion_mask

        snap_mask = self._completion_mask_through_first_eos(completion_ids)
        completion_texts = self._decode_completion_texts(completion_ids, snap_mask)
        rewards_binary = self._compute_binary_rewards(
            inputs,
            completion_texts,
            sample_count,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )
        if rewards_binary.numel() > 0:
            acc = float(self.accelerator.gather_for_metrics(rewards_binary.float()).mean().item())
        else:
            acc = 0.0
        self._log_metric("acc", acc)
        grouped = rewards_binary.view(-1, self.num_generations)
        all_correct_group = (grouped > 0.5).all(dim=1)
        all_wrong_group = (grouped < 0.5).all(dim=1)
        mixed_group = ~(all_correct_group | all_wrong_group)

        all_correct = all_correct_group.repeat_interleave(self.num_generations).unsqueeze(1)
        all_wrong = all_wrong_group.repeat_interleave(self.num_generations).unsqueeze(1)
        mixed = mixed_group.repeat_interleave(self.num_generations).unsqueeze(1)

        # Split mixed groups by correctness for diagnostics. Mixed samples keep
        # their original GRPO sequence advantage as the base A.
        mixed_base_adv = seq_advantages.unsqueeze(1)
        sample_correct = (rewards_binary > 0.5).unsqueeze(1)
        sample_wrong = ~sample_correct
        mixed_correct = mixed & sample_correct
        mixed_wrong = mixed & sample_wrong

        lambda_now = self._current_lambda()
        if abs(float(self.lmbda)) <= 1e-12:
            fallback_base_scale = 0.0
        else:
            fallback_base_scale = abs(float(lambda_now) / float(self.lmbda))
            fallback_base_scale = min(max(fallback_base_scale, 0.0), 1.0)

        def _shape_with_token_gap(
            base_adv: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            sign = torch.sign(base_adv)
            signed_gap = torch.clamp(sign * g, min=-20.0, max=20.0)
            is_positive_traj = sign >= 0
            raw_weight = torch.exp(signed_gap)
            clip_low_t = torch.where(
                is_positive_traj,
                torch.full_like(base_adv, float(self.correct_weight_clip_low)),
                torch.full_like(base_adv, float(self.wrong_weight_clip_low)),
            )
            clip_high_t = torch.where(
                is_positive_traj,
                torch.full_like(base_adv, float(self.correct_weight_clip_high)),
                torch.full_like(base_adv, float(self.wrong_weight_clip_high)),
            )
            weight = torch.minimum(torch.maximum(raw_weight, clip_low_t), clip_high_t)
            effective_delta = lambda_now * (weight - 1.0) * completion_mask
            factor = torch.clamp(1.0 + effective_delta, min=0.0)
            shaped = base_adv * factor

            flip_mask = (base_adv > 0) & (g < 0) & completion_mask.bool()
            down_weight = torch.clamp(
                torch.exp(torch.clamp(-g, min=-20.0, max=20.0)),
                min=max(1.0, float(self.wrong_weight_clip_low)),
                max=float(self.wrong_weight_clip_high),
            )
            down_factor = torch.clamp(1.0 + lambda_now * (down_weight - 1.0), min=0.0)
            flipped = -base_adv.abs() * down_factor
            shaped = torch.where(flip_mask, flipped, shaped)

            safe_base = torch.where(base_adv.abs() > 1e-12, base_adv, torch.ones_like(base_adv))
            effective_delta = torch.where(
                base_adv.abs() > 1e-12,
                (shaped / safe_base - 1.0) * completion_mask,
                torch.zeros_like(shaped),
            )
            shown_weight = torch.where(flip_mask, down_weight, weight)
            return shaped * completion_mask, shown_weight, effective_delta, flip_mask.float()

        mixed_adv, mixed_weight, mixed_delta, mixed_flip = _shape_with_token_gap(mixed_base_adv)
        mixed_mask = self._rollout_mask(seq_advantages).unsqueeze(1)
        mixed_fallback_adv = mixed_base_adv.expand_as(g) * completion_mask
        mixed_adv = torch.where(mixed_mask, mixed_adv, mixed_fallback_adv)
        mixed_delta = torch.where(mixed_mask, mixed_delta, torch.zeros_like(mixed_delta))
        mixed_flip = torch.where(mixed_mask, mixed_flip, torch.zeros_like(mixed_flip))

        all_correct_base_adv = (
            torch.full_like(g, float(self.all_correct_base_advantage) * fallback_base_scale)
            * completion_mask
        )
        all_wrong_base_adv = (
            torch.full_like(g, float(self.all_wrong_base_advantage) * fallback_base_scale)
            * completion_mask
        )
        all_correct_adv, correct_weight, correct_delta, correct_flip = _shape_with_token_gap(all_correct_base_adv)
        all_wrong_adv, wrong_weight, wrong_delta, wrong_flip = _shape_with_token_gap(all_wrong_base_adv)

        token_adv = torch.zeros_like(g)
        effective_delta = torch.zeros_like(g)
        flip_active = torch.zeros_like(g)
        if not self.strict_split_mixed_only:
            token_adv = torch.where(all_correct, all_correct_adv, token_adv)
            token_adv = torch.where(all_wrong, all_wrong_adv, token_adv)
            effective_delta = torch.where(all_correct, correct_delta, effective_delta)
            effective_delta = torch.where(all_wrong, wrong_delta, effective_delta)
            flip_active = torch.where(all_correct, correct_flip, flip_active)
            flip_active = torch.where(all_wrong, wrong_flip, flip_active)
        token_adv = torch.where(mixed, mixed_adv, token_adv)
        effective_delta = torch.where(mixed, mixed_delta, effective_delta)
        flip_active = torch.where(mixed, mixed_flip, flip_active)

        answer_weights = self._answer_weight_mask(
            completion_texts, completion_mask, decode_length_mask=snap_mask
        )
        token_adv = token_adv * answer_weights
        token_adv = torch.clamp(token_adv, min=self.adv_clip_low, max=self.adv_clip_high)
        token_adv = token_adv * completion_mask

        batch["advantages"] = token_adv

        # Per-group reward diagnostics.
        # Prompt-level grouping: [num_prompts, num_generations].
        # A group reward mean is computed over all completions that belong to prompts in that group.
        sample_all_correct = all_correct.squeeze(1)
        sample_all_wrong = all_wrong.squeeze(1)
        sample_mixed = mixed.squeeze(1)

        def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
            count = int(mask.sum().item())
            if count <= 0:
                return 0.0
            return float(values[mask].mean().item())

        prompt_count_all_correct = int(all_correct_group.sum().item())
        prompt_count_all_wrong = int(all_wrong_group.sum().item())
        prompt_count_mixed = int(mixed_group.sum().item())

        completion_count_all_correct = int(prompt_count_all_correct * self.num_generations)
        completion_count_all_wrong = int(prompt_count_all_wrong * self.num_generations)
        completion_count_mixed = int(prompt_count_mixed * self.num_generations)
        completion_count_mixed_correct = int((mixed_correct.squeeze(1)).sum().item())
        completion_count_mixed_wrong = int((mixed_wrong.squeeze(1)).sum().item())

        reward_mean_all_correct = _masked_mean(rewards_binary, sample_all_correct)
        reward_mean_all_wrong = _masked_mean(rewards_binary, sample_all_wrong)
        reward_mean_mixed = _masked_mean(rewards_binary, sample_mixed)
        if self.strict_split_mixed_only:
            no_feedback_group = all_correct_group | all_wrong_group
            feedback_group = mixed_group
        else:
            no_feedback_group = torch.zeros_like(mixed_group)
            feedback_group = torch.ones_like(mixed_group, dtype=torch.bool)

        token_count = completion_mask.sum().clamp(min=1.0)

        self._log_metric("token_gap_lambda", lambda_now)
        self._log_metric("mixed_only", float(self.strict_split_mixed_only))
        self._log_metric("feedback_group_frac", float(feedback_group.float().mean().item()))
        self._log_metric("no_feedback_group_frac", float(no_feedback_group.float().mean().item()))
        self._log_metric("correct_weight_clip_low", float(self.correct_weight_clip_low))
        self._log_metric("correct_weight_clip_high", float(self.correct_weight_clip_high))
        self._log_metric("wrong_weight_clip_low", float(self.wrong_weight_clip_low))
        self._log_metric("wrong_weight_clip_high", float(self.wrong_weight_clip_high))
        self._log_metric(
            "all_correct_base_advantage",
            float(self.all_correct_base_advantage),
        )
        self._log_metric(
            "all_wrong_base_advantage",
            float(self.all_wrong_base_advantage),
        )
        self._log_metric("group_all_correct_frac", float(all_correct_group.float().mean().item()))
        self._log_metric("group_all_wrong_frac", float(all_wrong_group.float().mean().item()))
        self._log_metric("group_mixed_frac", float(mixed_group.float().mean().item()))
        self._log_metric("reward_mean_all_correct", reward_mean_all_correct)
        self._log_metric("reward_mean_all_wrong", reward_mean_all_wrong)
        self._log_metric("reward_mean_mixed", reward_mean_mixed)
        self._log_metric("prompt_count_all_correct", float(prompt_count_all_correct))
        self._log_metric("prompt_count_all_wrong", float(prompt_count_all_wrong))
        self._log_metric("prompt_count_mixed", float(prompt_count_mixed))
        self._log_metric("completion_count_all_correct", float(completion_count_all_correct))
        self._log_metric("completion_count_all_wrong", float(completion_count_all_wrong))
        self._log_metric("completion_count_mixed", float(completion_count_mixed))
        self._log_metric("completion_count_mixed_correct", float(completion_count_mixed_correct))
        self._log_metric("completion_count_mixed_wrong", float(completion_count_mixed_wrong))
        self._log_metric(
            "effective_delta_pos_frac",
            float((((effective_delta > 0).float() * completion_mask).sum() / token_count).item()),
        )
        self._log_metric(
            "effective_delta_neg_frac",
            float((((effective_delta < 0).float() * completion_mask).sum() / token_count).item()),
        )
        self._log_metric(
            "effective_delta_zero_frac",
            float((((effective_delta == 0).float() * completion_mask).sum() / token_count).item()),
        )
        self._log_metric(
            "sign_flip_frac",
            float(((flip_active * completion_mask).sum() / token_count).item()),
        )
        self._log_metric("answer_weight_mean", float(answer_weights.mean().item()))
        self._log_metric("adv_abs_mean", float(((token_adv.abs() * completion_mask).sum() / token_count).item()))
        self._log_vector_stats("seq_adv", seq_advantages)
        self._log_masked_stats("token_gap", g, completion_mask)
        self._log_masked_stats("mixed_weight", mixed_weight, completion_mask)
        self._log_masked_stats("correct_weight", correct_weight, completion_mask)
        self._log_masked_stats("wrong_weight", wrong_weight, completion_mask)
        self._log_masked_stats("effective_delta", effective_delta, completion_mask)
        self._log_masked_stats("token_adv", token_adv, completion_mask)
        self._stash_rollout_for_checkpoint(
            inputs,
            completion_ids,
            completion_mask,
            reward_values=rewards_binary.detach().cpu().tolist(),
            seq_advantages_1d=seq_advantages,
            token_advantages=token_adv,
        )
        return batch
