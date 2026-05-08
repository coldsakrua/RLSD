from contextlib import nullcontext
from typing import Any, Dict, List, Sequence

import torch
from trl import GRPOTrainer


class RLSDTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        lmbda: float = 0.5,
        lmbda_decay_steps: int = 50,
        jsd_token_clip: float = 0.2,
        fixed_teacher: bool = False,
        rollout_filter: str = "all",
        teacher_prompt_template: str = (
            "{prompt}\n\n[Reference solution]\n{solution}\n\n[Student response]\n"
        ),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lmbda = float(lmbda)
        self.lmbda_decay_steps = int(lmbda_decay_steps)
        self.jsd_token_clip = float(jsd_token_clip)
        self.fixed_teacher = bool(fixed_teacher)
        self.rollout_filter = rollout_filter
        self.teacher_prompt_template = teacher_prompt_template

    def _current_lambda(self) -> float:
        if self.lmbda_decay_steps <= 0:
            return self.lmbda
        step = getattr(self.state, "global_step", 0)
        progress = min(max(step, 0), self.lmbda_decay_steps) / float(self.lmbda_decay_steps)
        return self.lmbda * (1.0 - progress)

    def _get_tokenizer(self):
        tokenizer = self.processing_class
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    def _prompt_to_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        tokenizer = self._get_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return str(prompt)

    def _extract_logps(self, output):
        if isinstance(output, tuple):
            return output[0]
        return output

    def _log_metric(self, key: str, value: float):
        mode = "train" if self.model.training else "eval"
        if mode not in self._metrics:
            self._metrics[mode] = {}
        if key not in self._metrics[mode]:
            self._metrics[mode][key] = []
        self._metrics[mode][key].append(value)

    def _build_teacher_prompts(self, inputs: Sequence[Dict[str, Any]]) -> List[str]:
        prompts: List[str] = []
        for row in inputs:
            prompt = self._prompt_to_text(row.get("prompt", ""))
            solution = row.get("solution", "")
            solution = solution if isinstance(solution, str) else str(solution)
            prompts.append(self.teacher_prompt_template.format(prompt=prompt, solution=solution))
        return prompts

    def _compute_student_logps(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        old_per_token_logps = batch.get("old_per_token_logps")
        if old_per_token_logps is not None:
            return old_per_token_logps.detach()

        input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        logits_to_keep = batch["completion_ids"].size(1)
        with torch.no_grad():
            output = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep,
            )
        return self._extract_logps(output).detach()

    def _compute_teacher_logps(
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
        teacher_input_ids = torch.cat([prefix_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([prefix_mask, completion_mask.long()], dim=1)
        logits_to_keep = completion_ids.size(1)

        model_for_teacher = self.model
        adapter_ctx = nullcontext()
        if self.fixed_teacher:
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "disable_adapter"):
                model_for_teacher = unwrapped
                adapter_ctx = unwrapped.disable_adapter()

        with torch.no_grad(), adapter_ctx:
            output = self._get_per_token_logps_and_entropies(
                model_for_teacher,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
            )
        return self._extract_logps(output).detach()

    def _rollout_mask(self, seq_advantages: torch.Tensor) -> torch.Tensor:
        if self.rollout_filter == "all":
            return torch.ones_like(seq_advantages, dtype=torch.bool)
        if self.rollout_filter == "positive":
            return seq_advantages > 0
        if self.rollout_filter == "negative":
            return seq_advantages < 0
        if self.rollout_filter == "mixed":
            if seq_advantages.numel() % self.num_generations != 0:
                return torch.ones_like(seq_advantages, dtype=torch.bool)
            grouped = seq_advantages.view(-1, self.num_generations)
            has_pos = (grouped > 0).any(dim=1)
            has_neg = (grouped < 0).any(dim=1)
            return (has_pos & has_neg).repeat_interleave(self.num_generations)
        raise ValueError(
            f"Unsupported rollout_filter={self.rollout_filter}. "
            "Choose from: all, positive, negative, mixed."
        )

    def _generate_and_score_completions(self, inputs):
        batch = super()._generate_and_score_completions(inputs)

        seq_advantages = batch["advantages"]
        if seq_advantages.dim() != 1:
            return batch

        completion_mask = batch["completion_mask"].float()
        completion_ids = batch["completion_ids"]
        student_logps = self._compute_student_logps(batch)
        teacher_prompts = self._build_teacher_prompts(inputs)
        teacher_logps = self._compute_teacher_logps(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            teacher_prompts=teacher_prompts,
        )

        lam = self._current_lambda()
        sign = torch.sign(seq_advantages).unsqueeze(1)
        token_delta = (teacher_logps - student_logps) * completion_mask
        token_weight = torch.exp(sign * token_delta)

        clip_low = 1.0 - self.jsd_token_clip
        clip_high = 1.0 + self.jsd_token_clip
        token_weight = torch.clamp(token_weight, min=clip_low, max=clip_high)

        uniform_adv = seq_advantages.unsqueeze(1)
        weighted_adv = uniform_adv * ((1.0 - lam) + lam * token_weight)

        rollout_mask = self._rollout_mask(seq_advantages).unsqueeze(1)
        token_advantages = torch.where(rollout_mask, weighted_adv, uniform_adv)
        token_advantages = token_advantages * completion_mask

        batch["advantages"] = token_advantages
        self._log_metric("rlsd_lambda", lam)
        self._log_metric("rlsd_w_mean", float(token_weight.mean().item()))
        return batch

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        output = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
        )
        if isinstance(output, tuple):
            per_token_logps, _ = output
        else:
            per_token_logps = output

        ref_per_token_logps = inputs.get("ref_per_token_logps")
        if ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        else:
            per_token_kl = None

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        elif advantages.dim() != 2:
            raise ValueError(f"advantages must be 1D or 2D, got shape={tuple(advantages.shape)}")

        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            # Keep a valid gradient path even when we don't cache old-policy logps
            # (common for single-iteration updates).
            old_per_token_logps = per_token_logps.detach()
        log_ratio = per_token_logps - old_per_token_logps

        importance_sampling_level = getattr(self.args, "importance_sampling_level", "token")
        if importance_sampling_level == "sequence":
            denom = completion_mask.sum(dim=1, keepdim=True).clamp(min=1)
            log_importance_weights = torch.sum(log_ratio * completion_mask, dim=1, keepdim=True) / denom
        elif importance_sampling_level == "sequence_token":
            denom = completion_mask.sum(dim=1, keepdim=True).clamp(min=1)
            seq_part = (torch.sum(log_ratio * completion_mask, dim=1, keepdim=True) / denom).detach()
            token_part = per_token_logps - per_token_logps.detach()
            log_importance_weights = seq_part + token_part
        else:
            log_importance_weights = log_ratio

        coef_1 = torch.exp(log_importance_weights)
        epsilon = getattr(self, "epsilon", getattr(self.args, "epsilon", 0.2))
        epsilon_high = getattr(self.args, "epsilon_high", None)
        if epsilon_high is None:
            coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
        else:
            low = torch.ones_like(coef_1) - epsilon
            high = torch.where(advantages < 0, 1 + epsilon, torch.ones_like(coef_1) + epsilon_high)
            coef_2 = torch.clamp(coef_1, low, high)

        delta = getattr(self.args, "delta", None)
        if delta is not None:
            coef_1 = torch.clamp(coef_1, max=delta)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        beta = getattr(self, "beta", getattr(self.args, "beta", 0.0))
        if per_token_kl is not None and beta != 0.0:
            per_token_loss = per_token_loss + beta * per_token_kl

        loss_type = getattr(self.args, "loss_type", "grpo")
        valid_token_count = torch.clamp(completion_mask.sum(), min=1)
        if loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / valid_token_count
        elif loss_type == "dr_grpo":
            max_len = completion_mask.size(1)
            loss = (per_token_loss * completion_mask).sum() / (completion_mask.size(0) * max_len)
        else:
            sample_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1))
            loss = sample_loss.mean()
            grad_acc_steps = max(
                1,
                getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps),
            )
            loss = loss / grad_acc_steps

        mode = "train" if self.model.training else "eval"
        if mode not in self._metrics:
            self._metrics[mode] = {}
        if "completion_length" not in self._metrics[mode]:
            self._metrics[mode]["completion_length"] = []
        self._metrics[mode]["completion_length"].append(
            float(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        )
        if per_token_kl is not None and "kl" not in self._metrics[mode]:
            self._metrics[mode]["kl"] = []
        if per_token_kl is not None:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
            self._metrics[mode]["kl"].append(float(self.accelerator.gather_for_metrics(mean_kl).mean().item()))

        clip_ratio = (torch.abs(coef_1 - coef_2) > 1e-6).float()
        if "clip_ratio" not in self._metrics[mode]:
            self._metrics[mode]["clip_ratio"] = []
        clip_ratio = (clip_ratio * completion_mask).sum() / valid_token_count
        self._metrics[mode]["clip_ratio"].append(
            float(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        )
        return loss
