from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import torch

from .prompt_utils import (
    DEFAULT_NO_REF_TEACHER_PROMPT,
    DEFAULT_OFFICIAL_TEACHER_PROMPT,
    DEFAULT_REF_TEACHER_PROMPT,
    DEFAULT_ROLLOUT_TEACHER_PROMPT,
    DEFAULT_TRANSITION_PROMPT,
    build_teacher_prompt,
    extract_prompt_text,
    extract_solution,
)
from .reward import compute_score

try:
    import ray
    from tensordict import TensorDict
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopManager,
        AgentLoopWorker,
        _InternalAgentLoopOutput,
    )
    from verl.utils import hf_tokenizer
except Exception:  # pragma: no cover - local syntax checks may not have veRL.
    ray = None
    TensorDict = None
    AgentLoopManager = object
    AgentLoopWorker = object
    _InternalAgentLoopOutput = object
    hf_tokenizer = None


def _cfg_get(config: Any, path: str, default: Any = None) -> Any:
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _row_from_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    row = {}
    for key, value in extra.items():
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        row[key] = value
    return row


class RLSDTeacherAgentLoopWorker(AgentLoopWorker):
    async def _compute_teacher_logprobs(self, output, sample_kwargs, validate: bool = False):
        # The default veRL worker computes teacher logprobs under the student prompt.
        # We attach aligned teacher-response logprobs after all samples in the batch
        # are available, so this hook is intentionally disabled here.
        return output

    def _teacher_prompt_mode(self) -> str:
        return str(_cfg_get(self.config, "algorithm.rlsd.teacher_prompt_mode", "reference_solution"))

    def _use_identical_student_prompt(self) -> bool:
        mode = self._teacher_prompt_mode().strip().lower()
        return mode in {"identical_student", "same_as_student", "student_identical"}

    def _teacher_logprob_response_length_cap(self) -> int:
        return int(_cfg_get(self.config, "algorithm.rlsd.teacher_logprob_response_length_cap", 0) or 0)

    def _student_prompt_ids_from_output(self, output: _InternalAgentLoopOutput) -> list[int]:
        prompt_ids = output.prompt_ids[0].detach().cpu().tolist()
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        while prompt_ids and prompt_ids[0] == pad_id:
            prompt_ids.pop(0)
        max_teacher_prompt_length = int(
            _cfg_get(self.config, "algorithm.rlsd.max_teacher_prompt_length", 0) or 0
        )
        if max_teacher_prompt_length > 0 and len(prompt_ids) > max_teacher_prompt_length:
            prompt_ids = prompt_ids[-max_teacher_prompt_length:]
        return prompt_ids

    def _teacher_prompt_templates(self) -> dict[str, str]:
        official = bool(_cfg_get(self.config, "algorithm.rlsd.official_teacher_prompt", False))
        return {
            "teacher_prompt_template": str(
                _cfg_get(
                    self.config,
                    "algorithm.rlsd.teacher_prompt_template",
                    DEFAULT_OFFICIAL_TEACHER_PROMPT if official else DEFAULT_REF_TEACHER_PROMPT,
                )
            ),
            "teacher_prompt_template_no_reference": str(
                _cfg_get(
                    self.config,
                    "algorithm.rlsd.teacher_prompt_template_no_reference",
                    DEFAULT_NO_REF_TEACHER_PROMPT,
                )
            ),
            "teacher_prompt_template_with_rollout": str(
                _cfg_get(
                    self.config,
                    "algorithm.rlsd.teacher_prompt_template_with_rollout",
                    DEFAULT_ROLLOUT_TEACHER_PROMPT,
                )
            ),
            "teacher_transition_prompt": str(
                _cfg_get(
                    self.config,
                    "algorithm.rlsd.teacher_transition_prompt",
                    DEFAULT_TRANSITION_PROMPT,
                )
            ),
        }

    def _decode_response(self, output: _InternalAgentLoopOutput) -> str:
        if self.tokenizer is None:
            return ""
        ids = output.response_ids[0]
        mask = output.response_mask[0].bool()
        valid = ids[mask].detach().cpu().tolist()
        return self.tokenizer.decode(valid, skip_special_tokens=True)

    def _row_for_output(self, output: _InternalAgentLoopOutput) -> dict[str, Any]:
        extra = getattr(output, "extra_fields", {}) or {}
        row = _row_from_extra(extra)
        row.setdefault("prompt", row.get("raw_prompt", ""))
        return row

    def _teacher_prompt_for_output(
        self,
        output: _InternalAgentLoopOutput,
        correct_rollout: str = "",
    ) -> str:
        row = self._row_for_output(output)
        templates = self._teacher_prompt_templates()
        return build_teacher_prompt(
            prompt=row.get("raw_prompt", row.get("prompt", "")),
            solution=extract_solution(row),
            mode=self._teacher_prompt_mode(),
            correct_rollout=correct_rollout,
            **templates,
        )

    async def _attach_teacher_for_output(
        self,
        output: _InternalAgentLoopOutput,
        teacher_prompt: str,
        validate: bool,
    ) -> None:
        if validate or not getattr(self, "distillation_enabled", False):
            return
        manager = getattr(self, "teacher_server_manager", None)
        if manager is None:
            return
        prompt_ids = output.prompt_ids[0]
        response_ids_padded = output.response_ids[0]
        response_mask = output.response_mask[0].bool()
        response_ids = response_ids_padded[response_mask].detach().cpu().tolist()
        response_cap = self._teacher_logprob_response_length_cap()
        if response_cap > 0:
            response_ids = response_ids[:response_cap]
        if not response_ids:
            return

        if self._use_identical_student_prompt():
            teacher_prompt_ids = self._student_prompt_ids_from_output(output)
        else:
            teacher_prompt_ids = self.tokenizer.encode(teacher_prompt, add_special_tokens=False)
            max_teacher_prompt_length = int(
                _cfg_get(self.config, "algorithm.rlsd.max_teacher_prompt_length", 0) or 0
            )
            if max_teacher_prompt_length > 0 and len(teacher_prompt_ids) > max_teacher_prompt_length:
                teacher_prompt_ids = teacher_prompt_ids[-max_teacher_prompt_length:]

        sequence_ids = teacher_prompt_ids + response_ids
        teacher_ids, teacher_logprobs = await manager.compute_teacher_logprobs_single(
            sequence_ids=sequence_ids,
            validate=validate,
        )
        teacher_ids_t = torch.as_tensor(teacher_ids)
        teacher_logprobs_t = torch.as_tensor(teacher_logprobs, dtype=torch.float32)
        if teacher_ids_t.dim() == 1:
            teacher_ids_t = teacher_ids_t.unsqueeze(-1)
        if teacher_logprobs_t.dim() == 1:
            teacher_logprobs_t = teacher_logprobs_t.unsqueeze(-1)
        teacher_ids_t = teacher_ids_t[-len(response_ids) :]
        teacher_logprobs_t = teacher_logprobs_t[-len(response_ids) :]

        total_width = output.prompt_ids.shape[1] + output.response_ids.shape[1]
        topk_width = int(teacher_logprobs_t.shape[-1])
        aligned_ids = torch.full(
            (1, total_width, topk_width),
            int(getattr(self.tokenizer, "pad_token_id", 0) or 0),
            dtype=torch.long,
        )
        aligned_logprobs = torch.zeros((1, total_width, topk_width), dtype=torch.float32)
        start = int(output.prompt_ids.shape[1])
        end = min(start + len(response_ids), total_width)
        valid_len = end - start
        if valid_len > 0:
            aligned_ids[0, start:end] = teacher_ids_t[:valid_len]
            aligned_logprobs[0, start:end] = teacher_logprobs_t[:valid_len]
        output.teacher_ids = aligned_ids
        output.teacher_logprobs = aligned_logprobs

    async def _attach_teacher_outputs(
        self,
        outputs: list[_InternalAgentLoopOutput],
        validate: bool,
    ) -> None:
        if validate or not getattr(self, "distillation_enabled", False):
            return
        mode = self._teacher_prompt_mode().strip().lower()
        correct_rollouts: dict[Any, str] = {}
        if mode in {"successful_rollout", "rollout"}:
            grouped: dict[Any, list[int]] = defaultdict(list)
            for i, output in enumerate(outputs):
                uid = (getattr(output, "extra_fields", {}) or {}).get("uid", i)
                if isinstance(uid, np.ndarray) and uid.shape == ():
                    uid = uid.item()
                grouped[uid].append(i)
            response_texts = [self._decode_response(output) for output in outputs]
            for _, idxs in grouped.items():
                success_idxs: list[int] = []
                for idx in idxs:
                    row = self._row_for_output(outputs[idx])
                    gt = extract_solution(row)
                    if compute_score(solution_str=response_texts[idx], ground_truth=gt) > 0.5:
                        success_idxs.append(idx)
                for idx in idxs:
                    peers = [peer for peer in success_idxs if peer != idx]
                    chosen_idx = peers[0] if peers else (success_idxs[0] if success_idxs else None)
                    correct_rollouts[idx] = "" if chosen_idx is None else response_texts[chosen_idx]

        tasks = []
        for i, output in enumerate(outputs):
            prompt = self._teacher_prompt_for_output(
                output,
                correct_rollout=correct_rollouts.get(i, ""),
            )
            tasks.append(self._attach_teacher_for_output(output, prompt, validate))
        if tasks:
            await asyncio.gather(*tasks)

    async def generate_sequences(self, batch):
        # This mirrors the upstream AgentLoopWorker flow and inserts one hook
        # after all rollouts are collected so RLRT can use a successful peer
        # completion as teacher context.
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(config.multi_turn.sampling_params)
        if batch.meta_info.get("validate", False):
            sampling_params["temperature"] = 0
            sampling_params["top_p"] = 1
            sampling_params["top_k"] = -1
            sampling_params["n"] = 1
        elif "n" not in sampling_params:
            sampling_params["n"] = config.n

        batch_size = len(batch.batch["raw_prompt_ids"])
        tasks = []
        for i in range(batch_size):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["request_id"] = batch.non_tensor_batch["uid"][i]
            kwargs["validate"] = batch.meta_info.get("validate", False)
            kwargs["sampling_params"] = sampling_params
            task = self._run_agent_loop(kwargs)
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)
        await self._attach_teacher_outputs(outputs, validate=batch.meta_info.get("validate", False))
        return self._postprocess(
            inputs=outputs,
            input_non_tensor_batch=batch.non_tensor_batch,
            validate=batch.meta_info.get("validate", False),
        )


class RLSDTeacherAgentLoopManager(AgentLoopManager):
    def __init__(self, *args: Any, **kwargs: Any):
        if ray is not None:
            self.agent_loop_workers_class = ray.remote(RLSDTeacherAgentLoopWorker)
        super().__init__(*args, **kwargs)
