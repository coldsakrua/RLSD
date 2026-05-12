from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence

import torch
from trl import GRPOTrainer

from data_utils import extract_last_user_text, normalize_prompt_to_standard_instruction
from reward_fn import extract_math_reward_answer


class RLSDTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        lmbda: float = 0.5,
        lmbda_decay_steps: int = 50,
        jsd_token_clip: float = 0.2,
        fixed_teacher: bool = False,
        teacher_update_interval_steps: int = 10,
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
        self.teacher_update_interval_steps = max(0, int(teacher_update_interval_steps))
        self.rollout_filter = rollout_filter
        self.teacher_prompt_template = teacher_prompt_template
        self._last_rollout_snapshot: Optional[Dict[str, Any]] = None
        self._teacher_snapshot_step: int = -1
        self._teacher_snapshot_state: Optional[Dict[str, torch.Tensor]] = None

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

    def _completion_ended_with_eos(
        self, completion_ids: torch.Tensor, completion_mask: torch.Tensor
    ) -> List[bool]:
        """True if the last non-masked completion token is the tokenizer EOS (natural stop)."""
        tokenizer = self._get_tokenizer()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            return [True] * int(completion_ids.size(0))
        out: List[bool] = []
        for row_ids, row_mask in zip(completion_ids, completion_mask):
            valid = row_ids[row_mask.bool()]
            if valid.numel() == 0:
                out.append(False)
            else:
                out.append(int(valid[-1].item()) == int(eos_id))
        return out

    def _expand_column_for_completions(self, values: Sequence[Any], target_len: int) -> List[Any]:
        if not values:
            return [""] * target_len
        values = list(values)
        if len(values) == target_len:
            return values
        if target_len % len(values) == 0:
            r = target_len // len(values)
            return [v for v in values for _ in range(r)]
        return [values[i % len(values)] for i in range(target_len)]

    def _decode_completion_texts(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> List[str]:
        tokenizer = self._get_tokenizer()
        texts: List[str] = []
        for ids_row, mask_row in zip(completion_ids, completion_mask):
            valid_ids = ids_row[mask_row.bool()].tolist()
            texts.append(self._decode_non_thinking_content(valid_ids, tokenizer))
        return texts

    def _end_think_token_id(self, tokenizer) -> Optional[int]:
        """Best-effort lookup for ``</think>`` token id (Qwen-style thinking template)."""
        try:
            tok_id = tokenizer.convert_tokens_to_ids("</think>")
        except Exception:
            return None
        if tok_id is None:
            return None
        try:
            tok_id = int(tok_id)
        except Exception:
            return None
        if tok_id < 0:
            return None
        return tok_id

    def _decode_non_thinking_content(self, valid_ids: List[int], tokenizer) -> str:
        """
        Match official Qwen-style parsing: if ``</think>`` appears, keep only tokens after its
        last occurrence; otherwise keep full decoded text.
        """
        if not valid_ids:
            return ""

        end_think_id = self._end_think_token_id(tokenizer)
        if end_think_id is not None:
            for i in range(len(valid_ids) - 1, -1, -1):
                if int(valid_ids[i]) == end_think_id:
                    tail = valid_ids[i + 1 :]
                    return tokenizer.decode(tail, skip_special_tokens=True).strip("\n")

        text = tokenizer.decode(valid_ids, skip_special_tokens=True)
        if "</think>" in text:
            return text.rsplit("</think>", 1)[-1].strip("\n")
        return text

    def _completion_mask_through_first_eos(self, completion_ids: torch.Tensor) -> torch.Tensor:
        """
        Mask through the first EOS (inclusive), else keep all completion positions.
        Matches TRL pre-``mask_truncated_completions`` semantics so snapshots/logs see the full rollout string.
        """
        tokenizer = self._get_tokenizer()
        device = completion_ids.device
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            return torch.ones_like(completion_ids, dtype=torch.long)
        is_eos = completion_ids == int(eos_id)
        seq_len = int(is_eos.size(1))
        eos_idx = torch.full((is_eos.size(0),), seq_len, dtype=torch.long, device=device)
        any_eos = is_eos.any(dim=1)
        eos_idx[any_eos] = is_eos.int().argmax(dim=1)[any_eos]
        seq = torch.arange(seq_len, device=device).unsqueeze(0).expand(is_eos.size(0), -1)
        return (seq <= eos_idx.unsqueeze(1)).long()

    def _decode_completion_texts_snapshot(self, completion_ids: torch.Tensor) -> List[str]:
        snap = self._completion_mask_through_first_eos(completion_ids)
        return self._decode_completion_texts(completion_ids, snap)

    def _stash_rollout_for_checkpoint(
        self,
        inputs: Sequence[Dict[str, Any]],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        reward_values: Optional[Sequence[float]] = None,
        seq_advantages_1d: Optional[torch.Tensor] = None,
        token_advantages: Optional[torch.Tensor] = None,
    ) -> None:
        if not getattr(self.args, "save_rollout_snapshots", True):
            return
        if hasattr(self, "accelerator") and not self.accelerator.is_main_process:
            return
        sample_count = int(completion_ids.size(0))
        snap_mask = self._completion_mask_through_first_eos(completion_ids)
        completions = self._decode_completion_texts(completion_ids, snap_mask)
        prompt_rows = [self._prompt_to_text(x.get("prompt", "")) for x in inputs]
        solution_rows = [x.get("solution", "") for x in inputs]
        prompts_exp = self._expand_column_for_completions(prompt_rows, sample_count)
        sols_exp = self._expand_column_for_completions(solution_rows, sample_count)
        ended_eos = self._completion_ended_with_eos(completion_ids, snap_mask)

        grpo_adv_list = None
        if seq_advantages_1d is not None and seq_advantages_1d.dim() == 1:
            grpo_adv_list = seq_advantages_1d.detach().cpu().tolist()

        token_adv_mean_list = None
        if token_advantages is not None and token_advantages.dim() == 2:
            denom = completion_mask.sum(dim=1).clamp(min=1)
            token_adv_mean_list = ((token_advantages * completion_mask).sum(dim=1) / denom).detach().cpu().tolist()

        samples: List[Dict[str, Any]] = []
        for i in range(sample_count):
            sol_i = str(sols_exp[i]) if i < len(sols_exp) else ""
            comp_i = completions[i]
            item: Dict[str, Any] = {
                "prompt": prompts_exp[i] if i < len(prompts_exp) else "",
                "solution": sol_i,
                "completion": comp_i,
                "extracted_answer": extract_math_reward_answer(comp_i, for_ground_truth=False),
                "extracted_ground_truth": extract_math_reward_answer(sol_i, for_ground_truth=True),
                "ended_with_eos": ended_eos[i],
            }
            if reward_values is not None and i < len(reward_values):
                item["reward"] = float(reward_values[i])
            if grpo_adv_list is not None and i < len(grpo_adv_list):
                item["grpo_advantage"] = float(grpo_adv_list[i])
            if token_adv_mean_list is not None and i < len(token_adv_mean_list):
                item["token_advantage_mean"] = float(token_adv_mean_list[i])
            samples.append(item)

        payload: Dict[str, Any] = {
            "rollout_global_step": int(getattr(self.state, "global_step", -1)),
            "epoch": float(getattr(self.state, "epoch", 0.0) or 0.0),
            "num_generations": int(getattr(self, "num_generations", 1)),
            "samples": samples,
        }
        if reward_values is not None and sample_count > 0:
            n = min(len(reward_values), sample_count)
            if n > 0:
                payload["acc"] = float(sum(float(reward_values[i]) for i in range(n)) / n)
        self._last_rollout_snapshot = payload

    def _prompt_to_text(self, prompt: Any) -> str:
        text = extract_last_user_text(prompt)
        if not text:
            return text
        try:
            normalized = normalize_prompt_to_standard_instruction(text)
            if isinstance(normalized, str):
                return normalized
        except Exception:
            pass
        return text

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

    def _teacher_anchor_step(self) -> int:
        step = int(getattr(self.state, "global_step", 0) or 0)
        interval = max(1, self.teacher_update_interval_steps)
        return (step // interval) * interval

    def _unwrap_model_for_teacher(self):
        model = self.model
        if hasattr(self, "accelerator"):
            try:
                model = self.accelerator.unwrap_model(model)
            except Exception:
                pass
        return model

    def _select_teacher_snapshot_params(self, model) -> List[tuple[str, torch.nn.Parameter]]:
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if params:
            return params
        # Fallback: if grad flags were altered unexpectedly, at least snapshot LoRA params.
        return [(name, p) for name, p in model.named_parameters() if "lora_" in name.lower()]

    def _refresh_teacher_snapshot_if_needed(self, model) -> None:
        if self.teacher_update_interval_steps <= 0:
            return
        anchor = self._teacher_anchor_step()
        if self._teacher_snapshot_state is not None and self._teacher_snapshot_step == anchor:
            return
        param_items = self._select_teacher_snapshot_params(model)
        snapshot: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in param_items:
                snapshot[name] = param.detach().cpu().clone()
        self._teacher_snapshot_state = snapshot
        self._teacher_snapshot_step = anchor

    def _teacher_forward_with_periodic_snapshot(
        self,
        model_for_teacher,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ):
        if self.teacher_update_interval_steps <= 0:
            adapter_ctx = nullcontext()
            if self.fixed_teacher and hasattr(model_for_teacher, "disable_adapter"):
                adapter_ctx = model_for_teacher.disable_adapter()
            with torch.no_grad(), adapter_ctx:
                return self._get_per_token_logps_and_entropies(
                    model_for_teacher,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                )

        self._refresh_teacher_snapshot_if_needed(model_for_teacher)
        snapshot = self._teacher_snapshot_state or {}
        if not snapshot:
            with torch.no_grad():
                return self._get_per_token_logps_and_entropies(
                    model_for_teacher,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                )

        param_by_name = dict(model_for_teacher.named_parameters())
        restore_state: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, teacher_tensor in snapshot.items():
                param = param_by_name.get(name)
                if param is None:
                    continue
                restore_state[name] = param.detach().clone()
                param.copy_(teacher_tensor.to(device=param.device, dtype=param.dtype))
            try:
                return self._get_per_token_logps_and_entropies(
                    model_for_teacher,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                )
            finally:
                for name, student_tensor in restore_state.items():
                    param = param_by_name.get(name)
                    if param is not None:
                        param.copy_(student_tensor)

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

        model_for_teacher = self._unwrap_model_for_teacher()
        output = self._teacher_forward_with_periodic_snapshot(
            model_for_teacher=model_for_teacher,
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            logits_to_keep=logits_to_keep,
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
        self._stash_rollout_for_checkpoint(
            inputs,
            completion_ids,
            completion_mask,
            reward_values=None,
            seq_advantages_1d=seq_advantages,
            token_advantages=token_advantages,
        )
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
