from typing import Tuple

import torch
from trl import GRPOTrainer

from rlsd_sign_flip_strict_split_trainer import RLSDSignFlipStrictSplitTrainer


class RLSDSignFlipWrongBoostStrictSplitTrainer(RLSDSignFlipStrictSplitTrainer):
    """
    Ablation on top of strict-split sign-flip OPSD:
    - keeps correct-path flip: base_adv > 0 and g < 0 -> negative advantage
    - adds wrong-path flip: base_adv < 0 and g > 0 (teacher prefers higher prob) -> positive advantage
    - optional strict_split_grpo_mixed_only_after_decay: before token_gap decay, full OPSD on all
      groups; after decay (lambda=0), skip teacher token-gap and update mixed groups with GRPO only.
    """

    def __init__(
        self,
        *args,
        strict_split_grpo_mixed_only_after_decay: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strict_split_grpo_mixed_only_after_decay = bool(strict_split_grpo_mixed_only_after_decay)

    def _in_grpo_mixed_only_phase(self, lambda_now: float) -> bool:
        return bool(
            self.strict_split_grpo_mixed_only_after_decay
            and self.lmbda_decay_steps > 0
            and lambda_now <= 1e-12
        )

    def _effective_mixed_only(self, lambda_now: float) -> bool:
        return bool(self.strict_split_mixed_only or self._in_grpo_mixed_only_phase(lambda_now))

    def _shape_with_token_gap_wrong_boost(
        self,
        base_adv: torch.Tensor,
        g: torch.Tensor,
        completion_mask: torch.Tensor,
        lambda_now: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Correct / positive-base: student over-confident vs teacher -> flip negative.
        down_flip_mask = (base_adv > 0) & (g < 0) & completion_mask.bool()
        down_weight = torch.clamp(
            torch.exp(torch.clamp(-g, min=-20.0, max=20.0)),
            min=max(1.0, float(self.wrong_weight_clip_low)),
            max=float(self.wrong_weight_clip_high),
        )
        down_factor = torch.clamp(1.0 + lambda_now * (down_weight - 1.0), min=0.0)
        flipped_down = -base_adv.abs() * down_factor
        shaped = torch.where(down_flip_mask, flipped_down, shaped)

        # Wrong / negative-base: teacher wants higher prob (g > 0) -> flip positive.
        up_flip_mask = (base_adv < 0) & (g > 0) & completion_mask.bool()
        up_weight = torch.clamp(
            torch.exp(torch.clamp(g, min=-20.0, max=20.0)),
            min=max(1.0, float(self.correct_weight_clip_low)),
            max=float(self.correct_weight_clip_high),
        )
        up_factor = torch.clamp(1.0 + lambda_now * (up_weight - 1.0), min=0.0)
        flipped_up = base_adv.abs() * up_factor
        shaped = torch.where(up_flip_mask, flipped_up, shaped)
        shaped = self._blend_teacher_shaping_length_cap(base_adv, shaped, completion_mask)

        flip_mask = down_flip_mask | up_flip_mask
        shown_weight = torch.where(down_flip_mask, down_weight, weight)
        shown_weight = torch.where(up_flip_mask, up_weight, shown_weight)

        safe_base = torch.where(base_adv.abs() > 1e-12, base_adv, torch.ones_like(base_adv))
        effective_delta = torch.where(
            base_adv.abs() > 1e-12,
            (shaped / safe_base - 1.0) * completion_mask,
            torch.zeros_like(shaped),
        )
        return (
            shaped * completion_mask,
            shown_weight,
            effective_delta,
            flip_mask.float(),
            up_flip_mask.float(),
        )

    def _generate_and_score_completions(self, inputs):
        batch = GRPOTrainer._generate_and_score_completions(self, inputs)

        seq_advantages = batch["advantages"]
        if seq_advantages.dim() != 1:
            return batch

        completion_mask = batch["completion_mask"].float()
        completion_ids = batch["completion_ids"]
        sample_count = seq_advantages.numel()
        if sample_count == 0 or sample_count % self.num_generations != 0:
            return batch

        student_logps = self._compute_student_logps(batch)
        lambda_now = self._current_lambda()
        grpo_mixed_only_phase = self._in_grpo_mixed_only_phase(lambda_now)
        mixed_only_now = self._effective_mixed_only(lambda_now)

        if grpo_mixed_only_phase:
            g = torch.zeros_like(student_logps)
        else:
            teacher_prompts = self._expand_to_samples(self._build_teacher_prompts(inputs), sample_count)
            teacher_logps = self._compute_teacher_logps_strict(
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                teacher_prompts=teacher_prompts,
            )
            g = (teacher_logps - student_logps).detach() * completion_mask
            g = self._mask_token_gap_within_teacher_cap(g, completion_mask)

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

        mixed_base_adv = seq_advantages.unsqueeze(1)
        sample_correct = (rewards_binary > 0.5).unsqueeze(1)
        sample_wrong = ~sample_correct
        mixed_correct = mixed & sample_correct
        mixed_wrong = mixed & sample_wrong

        if abs(float(self.lmbda)) <= 1e-12:
            fallback_base_scale = 0.0
        else:
            fallback_base_scale = abs(float(lambda_now) / float(self.lmbda))
            fallback_base_scale = min(max(fallback_base_scale, 0.0), 1.0)

        mixed_mask = self._rollout_mask(seq_advantages).unsqueeze(1)
        if grpo_mixed_only_phase:
            mixed_adv = mixed_base_adv.expand_as(g) * completion_mask
            mixed_weight = torch.ones_like(g)
            mixed_delta = torch.zeros_like(g)
            mixed_flip = torch.zeros_like(g)
            mixed_up_flip = torch.zeros_like(g)
            mixed_adv = torch.where(mixed_mask, mixed_adv, torch.zeros_like(mixed_adv))
        else:
            mixed_adv, mixed_weight, mixed_delta, mixed_flip, mixed_up_flip = (
                self._shape_with_token_gap_wrong_boost(mixed_base_adv, g, completion_mask, lambda_now)
            )
            mixed_fallback_adv = mixed_base_adv.expand_as(g) * completion_mask
            mixed_adv = torch.where(mixed_mask, mixed_adv, mixed_fallback_adv)
            mixed_delta = torch.where(mixed_mask, mixed_delta, torch.zeros_like(mixed_delta))
            mixed_flip = torch.where(mixed_mask, mixed_flip, torch.zeros_like(mixed_flip))
            mixed_up_flip = torch.where(mixed_mask, mixed_up_flip, torch.zeros_like(mixed_up_flip))

        all_correct_base_adv = (
            torch.full_like(g, float(self.all_correct_base_advantage) * fallback_base_scale)
            * completion_mask
        )
        all_wrong_base_adv = (
            torch.full_like(g, float(self.all_wrong_base_advantage) * fallback_base_scale)
            * completion_mask
        )
        if grpo_mixed_only_phase:
            all_correct_adv = torch.zeros_like(g)
            all_wrong_adv = torch.zeros_like(g)
            correct_weight = torch.ones_like(g)
            wrong_weight = torch.ones_like(g)
            correct_delta = torch.zeros_like(g)
            wrong_delta = torch.zeros_like(g)
            correct_flip = torch.zeros_like(g)
            wrong_flip = torch.zeros_like(g)
            correct_up_flip = torch.zeros_like(g)
            wrong_up_flip = torch.zeros_like(g)
        else:
            all_correct_adv, correct_weight, correct_delta, correct_flip, correct_up_flip = (
                self._shape_with_token_gap_wrong_boost(all_correct_base_adv, g, completion_mask, lambda_now)
            )
            all_wrong_adv, wrong_weight, wrong_delta, wrong_flip, wrong_up_flip = (
                self._shape_with_token_gap_wrong_boost(all_wrong_base_adv, g, completion_mask, lambda_now)
            )

        token_adv = torch.zeros_like(g)
        effective_delta = torch.zeros_like(g)
        flip_active = torch.zeros_like(g)
        up_flip_active = torch.zeros_like(g)
        if not mixed_only_now:
            token_adv = torch.where(all_correct, all_correct_adv, token_adv)
            token_adv = torch.where(all_wrong, all_wrong_adv, token_adv)
            effective_delta = torch.where(all_correct, correct_delta, effective_delta)
            effective_delta = torch.where(all_wrong, wrong_delta, effective_delta)
            flip_active = torch.where(all_correct, correct_flip, flip_active)
            flip_active = torch.where(all_wrong, wrong_flip, flip_active)
            up_flip_active = torch.where(all_correct, correct_up_flip, up_flip_active)
            up_flip_active = torch.where(all_wrong, wrong_up_flip, up_flip_active)
        token_adv = torch.where(mixed, mixed_adv, token_adv)
        effective_delta = torch.where(mixed, mixed_delta, effective_delta)
        flip_active = torch.where(mixed, mixed_flip, flip_active)
        up_flip_active = torch.where(mixed, mixed_up_flip, up_flip_active)

        answer_weights = self._answer_weight_mask(
            completion_texts, completion_mask, decode_length_mask=snap_mask
        )
        token_adv = token_adv * answer_weights
        token_adv = torch.clamp(token_adv, min=self.adv_clip_low, max=self.adv_clip_high)
        token_adv = token_adv * completion_mask

        batch["advantages"] = token_adv

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
        if mixed_only_now:
            no_feedback_group = all_correct_group | all_wrong_group
            feedback_group = mixed_group
        else:
            no_feedback_group = torch.zeros_like(mixed_group)
            feedback_group = torch.ones_like(mixed_group, dtype=torch.bool)

        token_count = completion_mask.sum().clamp(min=1.0)

        self._log_metric("token_gap_lambda", lambda_now)
        self._log_metric("teacher_shaping_length_cap", float(self.teacher_shaping_length_cap))
        self._log_metric(
            "teacher_logprob_response_length_cap",
            float(self._effective_teacher_logprob_response_length_cap()),
        )
        self._log_metric("mixed_only", float(mixed_only_now))
        self._log_metric("grpo_mixed_only_phase", float(grpo_mixed_only_phase))
        self._log_metric(
            "strict_split_grpo_mixed_only_after_decay",
            float(self.strict_split_grpo_mixed_only_after_decay),
        )
        self._log_metric("wrong_boost_flip", 1.0)
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
        self._log_metric(
            "wrong_positive_flip_frac",
            float(((up_flip_active * completion_mask).sum() / token_count).item()),
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
        return batch
