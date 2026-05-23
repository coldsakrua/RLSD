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

DEFAULT_RLRT_TEACHER_PROMPT_WITH_ROLLOUT = (
    "{prompt}\n\n[Correct rollout]\n{correct_rollout}\n\n[Student response]\n"
)


class RLRTTrainer(RLSDTrainer):
    """
    RLRT (RLVR with Reversed Teacher) token reweighting.

    For a token in a correct rollout, RLRT uses
      D_hat = log P_student(y_t) - log P_teacher(y_t)
      w_t = exp(sign(A) * D_hat)
      A_t = A * [(1 - lambda) + lambda * clip(w_t, 1 - eps_w, 1 + eps_w)].

    Incorrect rollouts keep their vanilla GRPO sequence advantage. The teacher
    context follows the RLRT paper's "correct rollout" setting by default: a
    successful rollout from the same prompt group is used as privileged context.
    """

    def __init__(
        self,
        *args,
        rlrt_weight_clip: float = 1.0,
        rlrt_teacher_context_mode: str = "successful_rollout",
        rlrt_teacher_prompt_template_with_rollout: str = DEFAULT_RLRT_TEACHER_PROMPT_WITH_ROLLOUT,
        adv_clip_low: float = -1.0e9,
        adv_clip_high: float = 1.0e9,
        answer_token_downweight: float = 1.0,
        suppress_gt_shortcut: bool = True,
        reward_binary_threshold: float = 0.5,
        fallback_tail_tokens: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rlrt_weight_clip = float(rlrt_weight_clip)
        self.rlrt_teacher_context_mode = str(rlrt_teacher_context_mode).strip().lower()
        allowed_modes = {"successful_rollout", "reference_solution"}
        if self.rlrt_teacher_context_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported rlrt_teacher_context_mode={rlrt_teacher_context_mode!r}. "
                f"Choose from: {sorted(allowed_modes)}."
            )
        self.rlrt_teacher_prompt_template_with_rollout = rlrt_teacher_prompt_template_with_rollout
        self.adv_clip_low = float(adv_clip_low)
        self.adv_clip_high = float(adv_clip_high)
        self.answer_token_downweight = float(answer_token_downweight)
        self.suppress_gt_shortcut = bool(suppress_gt_shortcut)
        self.reward_binary_threshold = float(reward_binary_threshold)
        self.fallback_tail_tokens = int(fallback_tail_tokens)

    def _expand_to_samples(self, values: Sequence[Any], target_len: int) -> List[Any]:
        if not values:
            return [""] * target_len
        values = list(values)
        if len(values) == target_len:
            return values
        if target_len % len(values) == 0:
            repeat = target_len // len(values)
            return [value for value in values for _ in range(repeat)]
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
        rewards = verifiable_math_reward(completions, solutions)

        reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        if reward_t.numel() != sample_count:
            reward_t = torch.zeros(sample_count, dtype=torch.float32, device=device)
        return (reward_t > self.reward_binary_threshold).float()

    def _answer_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for match in _ANSWER_TAG_RE.finditer(text):
            spans.append((match.start(), match.end()))
        for match in _BOXED_RE.finditer(text):
            spans.append((match.start(), match.end()))
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

    def _build_reference_solution_teacher_prompts(
        self,
        inputs: Sequence[Dict[str, Any]],
        sample_count: int,
    ) -> List[str]:
        prompt_texts = [self._prompt_to_text(row.get("prompt", "")) for row in inputs]
        solutions = [row.get("solution", "") for row in inputs]
        solutions = [s if isinstance(s, str) else str(s) for s in solutions]
        prompt_texts = self._expand_to_samples(prompt_texts, sample_count)
        solutions = self._expand_to_samples(solutions, sample_count)
        return [
            self.teacher_prompt_template.format(prompt=prompt, solution=solution)
            for prompt, solution in zip(prompt_texts, solutions)
        ]

    def _build_successful_rollout_teacher_prompts(
        self,
        inputs: Sequence[Dict[str, Any]],
        completion_texts: List[str],
        rewards_binary: torch.Tensor,
        sample_count: int,
    ) -> List[str]:
        if self.rlrt_teacher_context_mode == "reference_solution":
            return self._build_reference_solution_teacher_prompts(inputs, sample_count)

        prompt_texts = [self._prompt_to_text(row.get("prompt", "")) for row in inputs]
        if not prompt_texts:
            prompt_texts = [""]

        prompts: List[str] = []
        group_count = sample_count // self.num_generations
        reward_cpu = rewards_binary.detach().cpu()
        for group_idx in range(group_count):
            prompt = prompt_texts[group_idx % len(prompt_texts)]
            start = group_idx * self.num_generations
            correct_rel = [
                rel
                for rel in range(self.num_generations)
                if float(reward_cpu[start + rel].item()) > 0.5
            ]

            for rel in range(self.num_generations):
                current_idx = start + rel
                if correct_rel:
                    # Prefer a different successful peer when available; fall
                    # back to the current successful rollout if it is unique.
                    peers = [idx for idx in correct_rel if start + idx != current_idx]
                    chosen_rel = peers[0] if peers else correct_rel[0]
                    correct_rollout = completion_texts[start + chosen_rel]
                else:
                    # This prompt has no successful sampled rollout. RLRT's
                    # reward gate leaves these samples at vanilla GRPO, so this
                    # empty fallback only keeps teacher scoring well-defined.
                    correct_rollout = ""

                prompts.append(
                    self.rlrt_teacher_prompt_template_with_rollout.format(
                        prompt=prompt,
                        correct_rollout=correct_rollout,
                    )
                )
        if len(prompts) != sample_count:
            return prompts[:sample_count] + [""] * max(0, sample_count - len(prompts))
        return prompts

    def _compute_teacher_logps_rlrt(
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
        # Call GRPOTrainer directly to avoid RLSD's teacher-following weighting.
        batch = GRPOTrainer._generate_and_score_completions(self, inputs)

        seq_advantages = batch["advantages"]
        if seq_advantages.dim() != 1:
            return batch

        completion_mask = batch["completion_mask"].float()
        completion_ids = batch["completion_ids"]
        sample_count = seq_advantages.numel()
        if sample_count == 0 or sample_count % self.num_generations != 0:
            return batch

        snap_mask = self._completion_mask_through_first_eos(completion_ids)
        completion_texts = self._decode_completion_texts(completion_ids, snap_mask)
        rewards_binary = self._compute_binary_rewards(
            inputs,
            completion_texts,
            sample_count,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )

        student_logps = self._compute_student_logps(batch)
        teacher_prompts = self._build_successful_rollout_teacher_prompts(
            inputs=inputs,
            completion_texts=completion_texts,
            rewards_binary=rewards_binary,
            sample_count=sample_count,
        )
        teacher_logps = self._compute_teacher_logps_rlrt(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            teacher_prompts=teacher_prompts,
        )

        lambda_now = self._current_lambda()
        d_hat = (student_logps - teacher_logps).detach() * completion_mask
        sign = torch.sign(seq_advantages).unsqueeze(1)
        signed_gap = torch.clamp(sign * d_hat, min=-20.0, max=20.0)
        raw_weight = torch.exp(signed_gap)
        eps_w = max(0.0, float(self.rlrt_weight_clip))
        weight = torch.clamp(raw_weight, min=max(0.0, 1.0 - eps_w), max=1.0 + eps_w)

        uniform_adv = seq_advantages.unsqueeze(1).expand_as(d_hat) * completion_mask
        reweighted_adv = uniform_adv * ((1.0 - lambda_now) + lambda_now * weight)
        correct_mask = (rewards_binary > 0.5).unsqueeze(1)
        token_adv = torch.where(correct_mask, reweighted_adv, uniform_adv)

        answer_weights = self._answer_weight_mask(
            completion_texts, completion_mask, decode_length_mask=snap_mask
        )
        token_adv = token_adv * answer_weights
        token_adv = torch.clamp(token_adv, min=self.adv_clip_low, max=self.adv_clip_high)
        token_adv = token_adv * completion_mask
        batch["advantages"] = token_adv

        token_count = completion_mask.sum().clamp(min=1.0)
        if rewards_binary.numel() > 0:
            acc = self._reduce_scalar_mean(rewards_binary.float().mean())
        else:
            acc = 0.0
        correct_token_mask = correct_mask.float() * completion_mask
        correct_token_count = correct_token_mask.sum().clamp(min=1.0)

        self._log_metric("acc", acc)
        self._log_metric("rlrt_lambda", float(lambda_now))
        self._log_metric("rlrt_weight_clip", float(eps_w))
        self._log_metric("rlrt_correct_rollout_frac", float(rewards_binary.float().mean().item()))
        self._log_metric("rlrt_teacher_context_successful_rollout", float(self.rlrt_teacher_context_mode == "successful_rollout"))
        self._log_metric("answer_weight_mean", float(answer_weights.mean().item()))
        self._log_metric(
            "rlrt/active_token_frac",
            float((correct_token_mask.sum() / token_count).item()),
        )
        self._log_metric(
            "rlrt/weight_gt1_frac_correct",
            float((((weight > 1.0).float() * correct_token_mask).sum() / correct_token_count).item()),
        )
        self._log_vector_stats("seq_adv", seq_advantages)
        self._log_masked_stats("rlrt/d_hat", d_hat, completion_mask)
        self._log_masked_stats("rlrt/weight", weight, completion_mask)
        self._log_masked_stats("rlrt/weight_correct", weight, correct_token_mask)
        self._log_masked_stats("token_adv", token_adv, completion_mask)
        return batch
