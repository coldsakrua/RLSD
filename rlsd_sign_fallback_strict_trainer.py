import re
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence, Tuple

import torch
from trl import GRPOTrainer

from reward_fn import verifiable_math_reward
from rlsd_trainer import RLSDTrainer


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_TAIL_ANSWER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:answer|final answer)?\s*:?\s*([A-E]|[-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*$"
)


class RLSDSignFallbackStrictTrainer(RLSDTrainer):
    """
    Strict sign-constrained fallback RLSD:
    - mixed group: A_{i,t} = A_i * ((1-alpha) + alpha * w_{i,t}), sign from reward/advantage
    - all-correct group: positive-only fallback
    - all-wrong group: negative-only fallback
    """

    def __init__(
        self,
        *args,
        lambda_plus: float = 0.03,
        lambda_minus: float = 0.03,
        lambda_plus_min: float = 0.0,
        lambda_minus_min: float = 0.0,
        fallback_decay_steps: int = 200,
        fallback_eps0: float = 0.05,
        adv_clip_low: float = -1.0,
        adv_clip_high: float = 1.0,
        answer_token_downweight: float = 0.2,
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

    def _decode_completions(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> List[str]:
        tokenizer = self._get_tokenizer()
        texts: List[str] = []
        for ids_row, mask_row in zip(completion_ids, completion_mask):
            valid_ids = ids_row[mask_row.bool()].tolist()
            texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
        return texts

    def _compute_binary_rewards(self, inputs, completions: List[str], sample_count: int) -> torch.Tensor:
        device = self.accelerator.device
        solutions = self._expand_to_samples([x.get("solution", "") for x in inputs], sample_count)
        solutions = [s if isinstance(s, str) else str(s) for s in solutions]

        rewards = None
        reward_func = self.reward_funcs[0] if getattr(self, "reward_funcs", None) else None
        if callable(reward_func):
            try:
                rewards = reward_func(completions=completions, solution=solutions)
            except Exception:
                rewards = None
        if rewards is None:
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

    def _answer_weight_mask(self, completion_texts: List[str], completion_mask: torch.Tensor) -> torch.Tensor:
        n, max_len = completion_mask.shape
        device = completion_mask.device
        weights = torch.ones((n, max_len), dtype=torch.float32, device=device)

        if not self.suppress_gt_shortcut or self.answer_token_downweight >= 0.999:
            return weights

        down = float(self.answer_token_downweight)
        tokenizer = self._get_tokenizer()
        use_offset = getattr(tokenizer, "is_fast", False)

        for i, text in enumerate(completion_texts):
            valid_len = int(completion_mask[i].sum().item())
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
        tokenizer = self._get_tokenizer()
        device = completion_ids.device
        original_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            list(teacher_prompts),
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenizer.padding_side = original_padding_side

        prefix_ids = encoded["input_ids"].to(device)
        prefix_mask = encoded["attention_mask"].to(device)
        input_ids = torch.cat([prefix_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prefix_mask, completion_mask.long()], dim=1)
        logits_to_keep = completion_ids.size(1)

        model_for_teacher = self.model
        adapter_ctx = nullcontext()
        if self.fixed_teacher:
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "disable_adapter"):
                model_for_teacher = unwrapped
                adapter_ctx = unwrapped.disable_adapter()

        with torch.no_grad(), adapter_ctx:
            out = self._get_per_token_logps_and_entropies(
                model_for_teacher, input_ids, attention_mask, logits_to_keep
            )
        return out[0].detach() if isinstance(out, tuple) else out.detach()

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

        completion_texts = self._decode_completions(completion_ids, completion_mask)
        rewards_binary = self._compute_binary_rewards(inputs, completion_texts, sample_count)
        grouped = rewards_binary.view(-1, self.num_generations)
        all_correct_group = (grouped > 0.5).all(dim=1)
        all_wrong_group = (grouped < 0.5).all(dim=1)
        mixed_group = ~(all_correct_group | all_wrong_group)

        all_correct = all_correct_group.repeat_interleave(self.num_generations).unsqueeze(1)
        all_wrong = all_wrong_group.repeat_interleave(self.num_generations).unsqueeze(1)
        mixed = mixed_group.repeat_interleave(self.num_generations).unsqueeze(1)

        # mixed group: sign from reward-determined advantage; teacher scales magnitude only.
        base_adv = seq_advantages.unsqueeze(1)
        reward_sign = (2.0 * rewards_binary - 1.0).unsqueeze(1)
        adv_sign = torch.sign(base_adv)
        sign = torch.where(adv_sign == 0, reward_sign, adv_sign)
        w_mixed = torch.clamp(torch.exp(sign * g), min=clip_low, max=clip_high)
        alpha_mixed = self._current_lambda()
        mixed_adv = base_adv * ((1.0 - alpha_mixed) + alpha_mixed * w_mixed)
        mixed_mask = self._rollout_mask(seq_advantages).unsqueeze(1)
        mixed_adv = torch.where(mixed_mask, mixed_adv, base_adv)

        # all-correct: positive-only fallback
        w_plus = torch.clamp(torch.exp(g), min=clip_low, max=clip_high)
        plus_01 = self._rowwise_minmax_01(w_plus, completion_mask)
        plus_raw = (self.fallback_eps0 + plus_01) * completion_mask
        lambda_plus_now = self._current_fallback_lambda(self.lambda_plus, self.lambda_plus_min)
        plus_adv = self._normalize_mean_abs(plus_raw, completion_mask, lambda_plus_now)

        # all-wrong: negative-only fallback
        support = self._rowwise_minmax_01(torch.exp(torch.clamp(g, min=-20.0, max=20.0)), completion_mask)
        minus_raw = -(self.fallback_eps0 + (1.0 - support)) * completion_mask
        lambda_minus_now = self._current_fallback_lambda(self.lambda_minus, self.lambda_minus_min)
        minus_adv = self._normalize_mean_abs(minus_raw, completion_mask, lambda_minus_now)

        token_adv = torch.zeros_like(mixed_adv)
        token_adv = torch.where(mixed, mixed_adv, token_adv)
        token_adv = torch.where(all_correct, plus_adv, token_adv)
        token_adv = torch.where(all_wrong, minus_adv, token_adv)

        answer_weights = self._answer_weight_mask(completion_texts, completion_mask)
        token_adv = token_adv * answer_weights
        token_adv = torch.clamp(token_adv, min=self.adv_clip_low, max=self.adv_clip_high)
        token_adv = token_adv * completion_mask

        batch["advantages"] = token_adv

        self._log_metric("strict/mixed_alpha", alpha_mixed)
        self._log_metric("strict/lambda_plus", lambda_plus_now)
        self._log_metric("strict/lambda_minus", lambda_minus_now)
        self._log_metric("strict/group_all_correct_frac", float(all_correct_group.float().mean().item()))
        self._log_metric("strict/group_all_wrong_frac", float(all_wrong_group.float().mean().item()))
        self._log_metric("strict/group_mixed_frac", float(mixed_group.float().mean().item()))
        self._log_metric("strict/answer_weight_mean", float(answer_weights.mean().item()))
        self._log_metric("strict/adv_abs_mean", float((token_adv.abs() * completion_mask).sum().item() / completion_mask.sum().clamp(min=1).item()))
        return batch
