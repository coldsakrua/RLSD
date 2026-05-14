import math
from typing import Any, List, Optional, Sequence

import torch
from trl import GRPOTrainer

from reward_fn import verifiable_math_reward


class GRPOSplitMetricsTrainer(GRPOTrainer):
    """
    Plain GRPO trainer with strict_split-style correctness group diagnostics.

    This class does not change GRPO optimization behavior; it only logs
    additional metrics to align W&B panels with strict_split runs.
    """

    def __init__(self, *args, reward_binary_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_binary_threshold = float(reward_binary_threshold)

    def _get_tokenizer(self):
        tokenizer = self.processing_class
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    def _end_think_token_id(self, tokenizer) -> Optional[int]:
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

    def _decode_completion_texts(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> List[str]:
        tokenizer = self._get_tokenizer()
        texts: List[str] = []
        for ids_row, mask_row in zip(completion_ids, completion_mask):
            valid_ids = ids_row[mask_row.bool()].tolist()
            texts.append(self._decode_non_thinking_content(valid_ids, tokenizer))
        return texts

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

    def _compute_binary_rewards(self, inputs, completions: List[str], sample_count: int) -> torch.Tensor:
        device = self.accelerator.device
        solutions = self._expand_to_samples([x.get("solution", "") for x in inputs], sample_count)
        solutions = [s if isinstance(s, str) else str(s) for s in solutions]
        rewards = verifiable_math_reward(completions, solutions)

        reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        if reward_t.numel() != sample_count:
            reward_t = torch.zeros(sample_count, dtype=torch.float32, device=device)
        return (reward_t > self.reward_binary_threshold).float()

    def _completion_ended_with_eos(
        self, completion_ids: torch.Tensor, completion_mask: torch.Tensor
    ) -> List[bool]:
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

    def _reduce_scalar_mean(self, value: torch.Tensor | float) -> float:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), device=self.accelerator.device)
        value = value.detach().float().reshape(1)
        return float(self.accelerator.gather_for_metrics(value).mean().item())

    def _log_vector_stats(self, prefix: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        v = values.float()
        mean = self._reduce_scalar_mean(v.mean())
        sq_mean = self._reduce_scalar_mean((v * v).mean())
        std = math.sqrt(max(0.0, sq_mean - mean * mean))
        abs_mean = self._reduce_scalar_mean(v.abs().mean())
        pos_frac = self._reduce_scalar_mean((v > 0).float().mean())
        neg_frac = self._reduce_scalar_mean((v < 0).float().mean())
        zero_frac = self._reduce_scalar_mean((v == 0).float().mean())
        self._log_metric(f"{prefix}/mean", mean)
        self._log_metric(f"{prefix}/std", std)
        self._log_metric(f"{prefix}/abs_mean", abs_mean)
        self._log_metric(f"{prefix}/pos_frac", pos_frac)
        self._log_metric(f"{prefix}/neg_frac", neg_frac)
        self._log_metric(f"{prefix}/zero_frac", zero_frac)

    def _log_masked_stats(self, prefix: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        v = values.float()
        m = mask.float()
        denom = m.sum().clamp(min=1.0)
        mean = self._reduce_scalar_mean((v * m).sum() / denom)
        sq_mean = self._reduce_scalar_mean(((v * v) * m).sum() / denom)
        std = math.sqrt(max(0.0, sq_mean - mean * mean))
        abs_mean = self._reduce_scalar_mean((v.abs() * m).sum() / denom)
        pos_frac = self._reduce_scalar_mean(((v > 0).float() * m).sum() / denom)
        neg_frac = self._reduce_scalar_mean(((v < 0).float() * m).sum() / denom)
        zero_frac = self._reduce_scalar_mean(((v == 0).float() * m).sum() / denom)
        self._log_metric(f"{prefix}/mean", mean)
        self._log_metric(f"{prefix}/std", std)
        self._log_metric(f"{prefix}/abs_mean", abs_mean)
        self._log_metric(f"{prefix}/pos_frac", pos_frac)
        self._log_metric(f"{prefix}/neg_frac", neg_frac)
        self._log_metric(f"{prefix}/zero_frac", zero_frac)

    def _log_metric(self, key: str, value: float):
        if not hasattr(self, "_metrics") or self._metrics is None:
            self._metrics = {}
        mode = "train" if self.model.training else "eval"
        if mode not in self._metrics:
            self._metrics[mode] = {}
        if key not in self._metrics[mode]:
            self._metrics[mode][key] = []
        self._metrics[mode][key].append(value)

    def _generate_and_score_completions(self, inputs):
        batch = super()._generate_and_score_completions(inputs)

        completion_ids = batch.get("completion_ids")
        completion_mask = batch.get("completion_mask")
        seq_advantages = batch.get("advantages")
        if completion_ids is None or completion_mask is None or seq_advantages is None:
            return batch

        sample_count = int(completion_ids.size(0))
        if sample_count <= 0 or sample_count % self.num_generations != 0:
            return batch

        completion_mask_f = completion_mask.float()
        snap_mask = self._completion_mask_through_first_eos(completion_ids)
        completion_texts = self._decode_completion_texts(completion_ids, snap_mask)
        rewards_binary = self._compute_binary_rewards(inputs, completion_texts, sample_count)

        if rewards_binary.numel() > 0:
            acc = float(self.accelerator.gather_for_metrics(rewards_binary.float()).mean().item())
        else:
            acc = 0.0
        self._log_metric("acc", acc)

        grouped = rewards_binary.view(-1, self.num_generations)
        all_correct_group = (grouped > 0.5).all(dim=1)
        all_wrong_group = (grouped < 0.5).all(dim=1)
        mixed_group = ~(all_correct_group | all_wrong_group)

        all_correct = all_correct_group.repeat_interleave(self.num_generations)
        all_wrong = all_wrong_group.repeat_interleave(self.num_generations)
        mixed = mixed_group.repeat_interleave(self.num_generations)
        sample_correct = rewards_binary > 0.5
        sample_wrong = ~sample_correct
        mixed_correct = mixed & sample_correct
        mixed_wrong = mixed & sample_wrong

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
        completion_count_mixed_correct = int(mixed_correct.sum().item())
        completion_count_mixed_wrong = int(mixed_wrong.sum().item())

        reward_mean_all_correct = _masked_mean(rewards_binary, all_correct)
        reward_mean_all_wrong = _masked_mean(rewards_binary, all_wrong)
        reward_mean_mixed = _masked_mean(rewards_binary, mixed)
        reward_mean_mixed_correct = _masked_mean(rewards_binary, mixed_correct)
        reward_mean_mixed_wrong = _masked_mean(rewards_binary, mixed_wrong)

        self._log_metric("strict_split/group_all_correct_frac", float(all_correct_group.float().mean().item()))
        self._log_metric("strict_split/group_all_wrong_frac", float(all_wrong_group.float().mean().item()))
        self._log_metric("strict_split/group_mixed_frac", float(mixed_group.float().mean().item()))
        self._log_metric("strict_split/reward_mean_all_correct", reward_mean_all_correct)
        self._log_metric("strict_split/reward_mean_all_wrong", reward_mean_all_wrong)
        self._log_metric("strict_split/reward_mean_mixed", reward_mean_mixed)
        self._log_metric("strict_split/reward_mean_mixed_correct", reward_mean_mixed_correct)
        self._log_metric("strict_split/reward_mean_mixed_wrong", reward_mean_mixed_wrong)
        self._log_metric("strict_split/prompt_count_all_correct", float(prompt_count_all_correct))
        self._log_metric("strict_split/prompt_count_all_wrong", float(prompt_count_all_wrong))
        self._log_metric("strict_split/prompt_count_mixed", float(prompt_count_mixed))
        self._log_metric("strict_split/completion_count_all_correct", float(completion_count_all_correct))
        self._log_metric("strict_split/completion_count_all_wrong", float(completion_count_all_wrong))
        self._log_metric("strict_split/completion_count_mixed", float(completion_count_mixed))
        self._log_metric("strict_split/completion_count_mixed_correct", float(completion_count_mixed_correct))
        self._log_metric("strict_split/completion_count_mixed_wrong", float(completion_count_mixed_wrong))
        self._log_metric("strict_split/completion_correct_frac", float(sample_correct.float().mean().item()))
        self._log_metric("strict_split/completion_wrong_frac", float(sample_wrong.float().mean().item()))
        self._log_metric("strict_split/completion_mixed_correct_frac", float(mixed_correct.float().mean().item()))
        self._log_metric("strict_split/completion_mixed_wrong_frac", float(mixed_wrong.float().mean().item()))

        if seq_advantages.dim() == 1 and seq_advantages.numel() == sample_count:
            self._log_vector_stats("strict_split/seq_adv", seq_advantages)

            token_adv = seq_advantages.unsqueeze(1) * completion_mask_f
            token_count = completion_mask_f.sum().clamp(min=1.0)
            adv_abs_mean = float((token_adv.abs().sum() / token_count).item())
            self._log_metric("strict_split/adv_abs_mean", adv_abs_mean)
            self._log_masked_stats("strict_split/token_adv", token_adv, completion_mask_f)
        else:
            self._log_metric("strict_split/adv_abs_mean", 0.0)

        lens = completion_mask_f.sum(dim=1)
        effective_snap_mask = (snap_mask.float() * completion_mask_f).long()
        snap_lens = effective_snap_mask.float().sum(dim=1)
        ended_with_eos = self._completion_ended_with_eos(completion_ids, completion_mask)
        eos_frac = float(sum(1.0 for x in ended_with_eos if x) / max(1, len(ended_with_eos)))
        self._log_metric("strict_split/answer_weight_mean", 1.0)
        self._log_metric("strict_split/completion_len_mean", float(lens.mean().item()))
        self._log_metric("strict_split/completion_len_min", float(lens.min().item()))
        self._log_metric("strict_split/completion_len_max", float(lens.max().item()))
        self._log_metric("strict_split/terminated_len_mean", float(snap_lens.mean().item()))
        self._log_metric("strict_split/ended_with_eos_frac", eos_frac)
        self._log_metric("strict_split/no_eos_frac", 1.0 - eos_frac)
        return batch
