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


class RLSDSignFallbackStrictSplitTrainer(RLSDTrainer):
    """
    Strict split OPSD variant:
    - all-correct group + mixed-correct samples: positive OPSD log shaping
    - all-wrong group + mixed-wrong samples: negative OPSD log shaping
    - for wrong samples in mixed groups, keep only down-pressure
    """

    def __init__(
        self,
        *args,
        lambda_plus: float = 0.3,
        lambda_minus: float = 0.3,
        lambda_plus_min: float = 0.0,
        lambda_minus_min: float = 0.0,
        fallback_decay_steps: int = 50,
        fallback_eps0: float = 0.05,
        adv_clip_low: float = -1.0,
        adv_clip_high: float = 1.0,
        answer_token_downweight: float = 1.0,
        suppress_gt_shortcut: bool = True,
        reward_binary_threshold: float = 0.5,
        fallback_tail_tokens: int = 8,
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
        self.fallback_tail_tokens = int(fallback_tail_tokens)

    def _current_fallback_lambda(self, start: float, min_value: float) -> float:
        if self.fallback_decay_steps <= 0:
            return float(start)
        step = getattr(self.state, "global_step", 0)
        progress = min(max(step, 0), self.fallback_decay_steps) / float(self.fallback_decay_steps)
        return float(start) + (float(min_value) - float(start)) * progress

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

    def _rowwise_minmax_01(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.bool()
        row_min = torch.where(mask_bool, values, torch.full_like(values, float("inf"))).min(dim=1).values
        row_max = torch.where(mask_bool, values, torch.full_like(values, float("-inf"))).max(dim=1).values

        no_valid = mask_bool.sum(dim=1) == 0
        row_min = torch.where(no_valid, torch.zeros_like(row_min), row_min)
        row_max = torch.where(no_valid, torch.ones_like(row_max), row_max)

        denom = (row_max - row_min).clamp(min=1e-6).unsqueeze(1)
        out = (values - row_min.unsqueeze(1)) / denom
        return out * mask

    def _normalize_mean_abs(self, adv: torch.Tensor, mask: torch.Tensor, target_mean_abs: float) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1.0)
        mean_abs = (adv.abs() * mask).sum(dim=1) / lengths
        scale = (float(target_mean_abs) / mean_abs.clamp(min=1e-6)).unsqueeze(1)
        return adv * scale * mask

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

        clip_low = 1.0 - self.jsd_token_clip
        clip_high = 1.0 + self.jsd_token_clip

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

        # Split mixed groups by correctness.
        # One-sided keep rule:
        # - correct: keep only OPSD gains that increase token prob vs rollout (w_pos > 1)
        # - wrong: keep only OPSD gains that decrease token prob vs rollout (w_down > 1)
        base_adv = seq_advantages.unsqueeze(1)
        base_mag = base_adv.abs()
        sample_correct = (rewards_binary > 0.5).unsqueeze(1)
        sample_wrong = ~sample_correct
        mixed_correct = mixed & sample_correct
        mixed_wrong = mixed & sample_wrong

        alpha_mixed = self._current_lambda()
        w_pos = torch.clamp(torch.exp(g), min=clip_low, max=clip_high)
        w_down = torch.clamp(torch.exp(-g), min=clip_low, max=clip_high)
        gain_pos = torch.relu(w_pos - 1.0)
        gain_down = torch.relu(w_down - 1.0)

        mixed_pos_adv = base_mag * (1.0 + alpha_mixed * gain_pos)
        mixed_neg_adv = -base_mag * (1.0 + alpha_mixed * gain_down)
        mixed_mask = self._rollout_mask(seq_advantages).unsqueeze(1)
        mixed_pos_adv = torch.where(mixed_mask, mixed_pos_adv, base_adv)
        mixed_neg_adv = torch.where(mixed_mask, mixed_neg_adv, base_adv)

        # all-correct: one-sided positive OPSD (keep only w_pos > 1 gains).
        lambda_plus_now = self._current_fallback_lambda(self.lambda_plus, self.lambda_plus_min)
        plus_base = torch.full_like(g, float(self.fallback_eps0)) * completion_mask
        plus_adv = plus_base * (1.0 + lambda_plus_now * gain_pos)

        # all-wrong: one-sided negative OPSD (keep only w_down > 1 gains).
        lambda_minus_now = self._current_fallback_lambda(self.lambda_minus, self.lambda_minus_min)
        minus_base = -torch.full_like(g, float(self.fallback_eps0)) * completion_mask
        minus_adv = minus_base * (1.0 + lambda_minus_now * gain_down)

        token_adv = torch.zeros_like(base_adv)
        token_adv = torch.where(all_correct, plus_adv, token_adv)
        token_adv = torch.where(all_wrong, minus_adv, token_adv)
        token_adv = torch.where(mixed_correct, mixed_pos_adv, token_adv)
        token_adv = torch.where(mixed_wrong, mixed_neg_adv, token_adv)

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

        self._log_metric("strict_split/mixed_alpha", alpha_mixed)
        self._log_metric("strict_split/lambda_plus", lambda_plus_now)
        self._log_metric("strict_split/lambda_minus", lambda_minus_now)
        self._log_metric("strict_split/group_all_correct_frac", float(all_correct_group.float().mean().item()))
        self._log_metric("strict_split/group_all_wrong_frac", float(all_wrong_group.float().mean().item()))
        self._log_metric("strict_split/group_mixed_frac", float(mixed_group.float().mean().item()))
        self._log_metric("strict_split/reward_mean_all_correct", reward_mean_all_correct)
        self._log_metric("strict_split/reward_mean_all_wrong", reward_mean_all_wrong)
        self._log_metric("strict_split/reward_mean_mixed", reward_mean_mixed)
        self._log_metric("strict_split/prompt_count_all_correct", float(prompt_count_all_correct))
        self._log_metric("strict_split/prompt_count_all_wrong", float(prompt_count_all_wrong))
        self._log_metric("strict_split/prompt_count_mixed", float(prompt_count_mixed))
        self._log_metric("strict_split/completion_count_all_correct", float(completion_count_all_correct))
        self._log_metric("strict_split/completion_count_all_wrong", float(completion_count_all_wrong))
        self._log_metric("strict_split/completion_count_mixed", float(completion_count_mixed))
        self._log_metric("strict_split/completion_count_mixed_correct", float(completion_count_mixed_correct))
        self._log_metric("strict_split/completion_count_mixed_wrong", float(completion_count_mixed_wrong))
        self._log_metric(
            "strict_split/one_sided_gain_pos_frac",
            float(((gain_pos > 0).float() * completion_mask).sum().item() / completion_mask.sum().clamp(min=1).item()),
        )
        self._log_metric(
            "strict_split/one_sided_gain_down_frac",
            float(((gain_down > 0).float() * completion_mask).sum().item() / completion_mask.sum().clamp(min=1).item()),
        )
        self._log_metric("strict_split/answer_weight_mean", float(answer_weights.mean().item()))
        self._log_metric("strict_split/adv_abs_mean", float((token_adv.abs() * completion_mask).sum().item() / completion_mask.sum().clamp(min=1).item()))
        self._stash_rollout_for_checkpoint(
            inputs,
            completion_ids,
            completion_mask,
            reward_values=rewards_binary.detach().cpu().tolist(),
            seq_advantages_1d=seq_advantages,
            token_advantages=token_adv,
        )
        return batch
