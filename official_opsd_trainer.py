import inspect
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list, gather_object
from transformers import Trainer

from data_utils import (
    coerce_prompt_to_qwen3_user_messages,
    extract_last_user_text,
)
from run_logging import normalize_metric_key

try:
    from accelerate.utils import is_peft_model
except Exception:
    is_peft_model = None

try:
    from trl.extras.vllm_client import VLLMClient
except Exception:
    VLLMClient = None

try:
    from trl.models.utils import unwrap_model_for_generation
except Exception:
    unwrap_model_for_generation = None


DEFAULT_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step -- do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something does not work out:\n"
)

DEFAULT_OFFICIAL_TEACHER_PROMPT = (
    "Problem: {prompt}\n\n"
    "Here is a reference solution to this problem:\n"
    "=== Reference Solution Begin ===\n"
    "{solution}\n"
    "=== Reference Solution End ===\n"
    "{transition}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_peft_adapter(model: torch.nn.Module) -> bool:
    unwrapped = model
    if hasattr(model, "module"):
        unwrapped = model.module
    if is_peft_model is not None:
        try:
            return bool(is_peft_model(model) or is_peft_model(unwrapped))
        except Exception:
            pass
    return hasattr(unwrapped, "disable_adapter")


class OfficialOPSDDataCollator:
    """
    Build the two prompts used by official OPSD:
    - student prompt: the normal rollout prompt, without the reference solution
    - teacher prompt: the same problem plus the reference solution as privileged context

    The trainer performs on-policy generation inside ``training_step`` so the
    collator intentionally returns prompt strings rather than token tensors.
    """

    def __init__(
        self,
        tokenizer,
        *,
        student_prompt_as_chat: bool = True,
        student_thinking: bool = False,
        teacher_thinking: bool = False,
        teacher_prompt_template: str = DEFAULT_OFFICIAL_TEACHER_PROMPT,
        teacher_transition_prompt: str = DEFAULT_TRANSITION_PROMPT,
    ) -> None:
        self.tokenizer = tokenizer
        self.student_prompt_as_chat = bool(student_prompt_as_chat)
        self.student_thinking = bool(student_thinking)
        self.teacher_thinking = bool(teacher_thinking)
        self.teacher_prompt_template = teacher_prompt_template
        self.teacher_transition_prompt = teacher_transition_prompt

    def _apply_chat_template(
        self,
        messages: Any,
        *,
        enable_thinking: bool,
        add_generation_prompt: bool = True,
    ) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _student_prompt_text(self, prompt: Any) -> str:
        if isinstance(prompt, (list, dict)):
            messages = coerce_prompt_to_qwen3_user_messages(prompt)
            return self._apply_chat_template(
                messages,
                enable_thinking=self.student_thinking,
                add_generation_prompt=True,
            )

        text = prompt.strip() if isinstance(prompt, str) else str(prompt).strip()
        if self.student_prompt_as_chat:
            messages = [{"role": "user", "content": text}]
            return self._apply_chat_template(
                messages,
                enable_thinking=self.student_thinking,
                add_generation_prompt=True,
            )
        return text

    def _teacher_prompt_text(self, prompt: Any, solution: Any) -> str:
        problem = extract_last_user_text(prompt)
        if not problem:
            problem = prompt.strip() if isinstance(prompt, str) else str(prompt).strip()
        solution_text = solution if isinstance(solution, str) else str(solution)
        user_message = self.teacher_prompt_template.format(
            prompt=problem,
            problem=problem,
            solution=solution_text,
            transition=self.teacher_transition_prompt,
        )
        return self._apply_chat_template(
            [{"role": "user", "content": user_message}],
            enable_thinking=self.teacher_thinking,
            add_generation_prompt=True,
        )

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        student_prompt_texts: List[str] = []
        teacher_prompt_texts: List[str] = []
        solutions: List[str] = []
        raw_prompts: List[Any] = []

        for row in features:
            prompt = row.get("prompt", row.get("problem", ""))
            solution = row.get("solution", row.get("answer", ""))
            raw_prompts.append(prompt)
            solutions.append(solution if isinstance(solution, str) else str(solution))
            student_prompt_texts.append(self._student_prompt_text(prompt))
            teacher_prompt_texts.append(self._teacher_prompt_text(prompt, solution))

        return {
            "prompt": raw_prompts,
            "solution": solutions,
            "student_prompt_text": student_prompt_texts,
            "teacher_prompt_text": teacher_prompt_texts,
        }


class OfficialOPSDTrainer(Trainer):
    """
    OPSD implemented against the local transformers/TRL stack.

    Algorithmically this matches the public OPSD implementation's default path:
    sample one on-policy completion from the student prompt, score the same
    completion under a privileged teacher prompt containing the reference
    solution, and minimize full-vocabulary generalized JSD over generated tokens.
    """

    def __init__(
        self,
        *args,
        processing_class=None,
        max_student_prompt_length: Optional[int] = None,
        max_teacher_prompt_length: Optional[int] = None,
        max_completion_length: int = 1024,
        lmbda: float = 1.0,
        beta: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        generation_extra_kwargs: Optional[Dict[str, Any]] = None,
        fixed_teacher: bool = False,
        top_k_loss: Optional[int] = None,
        jsd_token_clip: Optional[float] = None,
        use_vllm: bool = False,
        vllm_guided_decoding_regex: Optional[str] = None,
        vllm_sync_frequency: int = 1,
        save_generation_steps: int = 0,
        **kwargs,
    ) -> None:
        trainer_init = inspect.signature(Trainer.__init__).parameters
        super_kwargs = dict(kwargs)
        if processing_class is not None:
            if "processing_class" in trainer_init:
                super_kwargs["processing_class"] = processing_class
            else:
                super_kwargs["tokenizer"] = processing_class
        super().__init__(*args, **super_kwargs)

        self.processing_class = processing_class if processing_class is not None else getattr(self, "tokenizer", None)
        self.max_student_prompt_length = max_student_prompt_length
        self.max_teacher_prompt_length = max_teacher_prompt_length
        self.max_completion_length = int(max_completion_length)
        self.lmbda = float(lmbda)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k) if top_k is not None else 0
        self.min_p = float(min_p)
        self.repetition_penalty = float(repetition_penalty)
        self.presence_penalty = float(presence_penalty)
        self.generation_extra_kwargs = dict(generation_extra_kwargs or {})
        self.fixed_teacher = bool(fixed_teacher)
        self.top_k_loss = int(top_k_loss) if top_k_loss and int(top_k_loss) > 0 else None
        self.jsd_token_clip = (
            float(jsd_token_clip) if jsd_token_clip is not None and float(jsd_token_clip) > 0 else None
        )
        self.use_vllm = bool(use_vllm)
        self.vllm_guided_decoding_regex = vllm_guided_decoding_regex
        self.vllm_sync_frequency = max(1, int(vllm_sync_frequency or 1))
        self.save_generation_steps = max(0, int(save_generation_steps or 0))
        self._last_vllm_sync_step = -1
        self._metrics: Dict[str, Dict[str, List[float]]] = {"train": {}, "eval": {}}
        self._generation_outputs_buffer: List[Dict[str, Any]] = []

        if self.fixed_teacher and not _has_peft_adapter(self.model):
            raise ValueError("fixed_teacher=True requires a PEFT/LoRA model so the teacher can disable adapters.")

        self.vllm_client = None
        if self.use_vllm:
            if VLLMClient is None:
                raise ImportError("trl.extras.vllm_client.VLLMClient is unavailable in this environment.")
            self._init_vllm_client()

    @staticmethod
    def generalized_jsd_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        beta: float = 0.5,
        temperature: float = 1.0,
        reduction: str = "batchmean",
        top_k: Optional[int] = None,
        token_clip: Optional[float] = None,
    ) -> torch.Tensor:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            k = min(int(top_k), int(teacher_logits.size(-1)))
            _, top_k_indices = torch.topk(teacher_logits, k=k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs + torch.log1p(-beta_t),
                        teacher_log_probs + torch.log(beta_t),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta_t * kl_teacher + (1.0 - beta_t) * kl_student

        if token_clip is not None:
            jsd = jsd.clamp(max=float(token_clip))

        mask = None
        if labels is not None:
            mask = labels != -100
            if not bool(mask.any()):
                return student_logits.sum() * 0.0
            jsd = jsd[mask]

        if reduction == "batchmean":
            if mask is not None:
                return jsd.sum() / mask.sum().clamp(min=1)
            return jsd.sum() / max(1, jsd.size(0))
        if reduction == "sum":
            return jsd.sum()
        if reduction == "mean":
            return jsd.mean()
        return jsd

    def _log_metric(self, key: str, value: float) -> None:
        key = normalize_metric_key(key)
        mode = "train" if self.model.training else "eval"
        self._metrics.setdefault(mode, {}).setdefault(key, []).append(float(value))

    def _reduce_scalar_mean(self, value: torch.Tensor | float) -> float:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), device=self.accelerator.device)
        value = value.detach().float().reshape(1)
        return float(self.accelerator.gather_for_metrics(value).mean().item())

    def _log_masked_stats(self, prefix: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.float()
        denom = m.sum().clamp(min=1.0)
        v = values.float()
        mean = self._reduce_scalar_mean((v * m).sum() / denom)
        sq_mean = self._reduce_scalar_mean(((v * v) * m).sum() / denom)
        std = math.sqrt(max(0.0, sq_mean - mean * mean))
        abs_mean = self._reduce_scalar_mean((v.abs() * m).sum() / denom)
        self._log_metric(f"{prefix}/mean", mean)
        self._log_metric(f"{prefix}/std", std)
        self._log_metric(f"{prefix}/abs_mean", abs_mean)

    def _init_vllm_client(self) -> None:
        if not self.accelerator.is_main_process:
            return

        host = getattr(self.args, "vllm_server_host", None)
        port = getattr(self.args, "vllm_server_port", None)
        base_url = getattr(self.args, "vllm_server_base_url", None)
        if base_url and (host is None or port is None):
            parsed = urlparse(base_url)
            host = host or parsed.hostname or "127.0.0.1"
            port = port or parsed.port or (443 if parsed.scheme == "https" else 80)
        host = host or "127.0.0.1"
        port = int(port or 8000)
        timeout = int(getattr(self.args, "vllm_server_timeout", 300) or 300)

        constructors = [
            dict(server_host=host, server_port=port, connection_timeout=timeout),
            dict(host=host, server_port=port, connection_timeout=timeout),
            dict(host=host, port=port, connection_timeout=timeout),
            dict(base_url=base_url, connection_timeout=timeout) if base_url else None,
        ]
        last_error = None
        for kwargs in constructors:
            if kwargs is None:
                continue
            try:
                self.vllm_client = VLLMClient(**kwargs)
                break
            except TypeError as exc:
                last_error = exc
        if self.vllm_client is None:
            raise TypeError(f"Could not initialize VLLMClient with local TRL API: {last_error}")
        if hasattr(self.vllm_client, "init_communicator"):
            self.vllm_client.init_communicator()

    def _sync_vllm_if_needed(self) -> None:
        if not self.use_vllm:
            return
        step = int(getattr(self.state, "global_step", 0) or 0)
        if step == self._last_vllm_sync_step:
            return
        if step != 0 and step % self.vllm_sync_frequency != 0:
            return
        self._move_model_to_vllm()
        self._last_vllm_sync_step = step

    def _move_model_to_vllm(self) -> None:
        self.accelerator.wait_for_everyone()
        model = self.accelerator.unwrap_model(self.model)
        prefix = getattr(model, "prefix", "")

        if _has_peft_adapter(model):
            if hasattr(model, "merge_adapter"):
                model.merge_adapter()
            try:
                for name, param in model.named_parameters():
                    send_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if prefix and prefix in send_name:
                        continue
                    if "original_module" in send_name:
                        continue
                    if "lora_" in send_name.lower():
                        continue
                    send_name = send_name.replace("modules_to_save.default.", "")
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(send_name, param.data)
            finally:
                if hasattr(model, "unmerge_adapter"):
                    model.unmerge_adapter()
        else:
            for name, param in model.named_parameters():
                if self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(name, param.data)

        if self.accelerator.is_main_process and hasattr(self.vllm_client, "reset_prefix_cache"):
            self.vllm_client.reset_prefix_cache()
        self.accelerator.wait_for_everyone()

    def _normalize_completion_ids(self, raw_outputs: Any) -> List[List[int]]:
        out: List[List[int]] = []
        for item in raw_outputs:
            if hasattr(item, "token_ids"):
                ids = getattr(item, "token_ids")
            elif isinstance(item, dict) and "token_ids" in item:
                ids = item["token_ids"]
            elif isinstance(item, list) and item and isinstance(item[0], list):
                ids = item[0]
            else:
                ids = item
            out.append([int(x) for x in list(ids)])
        return out

    def _call_vllm_generate(self, prompts: Sequence[str]) -> List[List[int]]:
        top_k = self.top_k if self.top_k and self.top_k > 0 else -1
        kwargs: Dict[str, Any] = {
            "prompts": list(prompts),
            "n": 1,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_completion_length,
            "presence_penalty": self.presence_penalty,
        }
        if self.vllm_guided_decoding_regex:
            kwargs["guided_decoding_regex"] = self.vllm_guided_decoding_regex
        kwargs.update(self.generation_extra_kwargs)

        unsupported: List[str] = []
        for _ in range(8):
            try:
                return self._normalize_completion_ids(self.vllm_client.generate(**kwargs))
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in list(kwargs):
                    if f"'{key}'" in message or f"{key}" in message:
                        if key not in ("prompts", "max_tokens"):
                            unsupported.append(key)
                            kwargs.pop(key, None)
                            removed = True
                            break
                if not removed:
                    raise
        raise TypeError(f"Could not call VLLMClient.generate; unsupported keys removed: {unsupported}")

    def _generate_vllm(self, student_prompt_texts: Sequence[str]) -> List[List[int]]:
        start_time = time.time()
        all_prompts = gather_object(list(student_prompt_texts))
        if self.accelerator.is_main_process:
            completion_ids = self._call_vllm_generate(all_prompts)
        else:
            completion_ids = [None] * len(all_prompts)
        broadcasted = broadcast_object_list(completion_ids, from_process=0)
        if broadcasted is not None:
            completion_ids = broadcasted

        per_rank = len(student_prompt_texts)
        start = self.accelerator.process_index * per_rank
        local = completion_ids[start : start + per_rank]
        total_tokens = sum(len(x) for x in local)
        elapsed = max(1e-6, time.time() - start_time)
        self._log_metric("opsd/generation_tokens_per_sec", total_tokens / elapsed)
        self._log_metric("opsd/completion_len_mean", total_tokens / max(1, len(local)))
        return local

    def _generate_torch(self, model: torch.nn.Module, student_prompt_texts: Sequence[str]) -> List[List[int]]:
        tokenizer = self.processing_class
        device = self.accelerator.device
        old_padding_side = getattr(tokenizer, "padding_side", "left")
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            list(student_prompt_texts),
            return_tensors="pt",
            padding=True,
            truncation=self.max_student_prompt_length is not None,
            max_length=self.max_student_prompt_length,
            add_special_tokens=False,
        ).to(device)
        tokenizer.padding_side = old_padding_side

        gen_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        if self.top_k and self.top_k > 0:
            gen_kwargs["top_k"] = self.top_k
        gen_kwargs.update(self.generation_extra_kwargs)

        original_use_cache = getattr(model.config, "use_cache", None)
        if original_use_cache is not None:
            model.config.use_cache = True
        try:
            with torch.no_grad():
                generated = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    **gen_kwargs,
                )
        finally:
            if original_use_cache is not None:
                model.config.use_cache = original_use_cache

        prompt_width = encoded["input_ids"].size(1)
        completions = generated[:, prompt_width:]
        out: List[List[int]] = []
        pad_id = tokenizer.pad_token_id
        for row in completions:
            ids = [int(x) for x in row.tolist()]
            while ids and pad_id is not None and ids[-1] == int(pad_id):
                ids.pop()
            out.append(ids)
        return out

    def _tokenize_prompts(
        self,
        texts: Sequence[str],
        *,
        max_length: Optional[int],
    ) -> List[List[int]]:
        encoded = self.processing_class(
            list(texts),
            padding=False,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
        )
        return [[int(x) for x in ids] for ids in encoded["input_ids"]]

    def _build_full_sequences(
        self,
        prompt_ids_list: Sequence[Sequence[int]],
        completion_ids_list: Sequence[Sequence[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_id = self.processing_class.pad_token_id
        if pad_id is None:
            pad_id = self.processing_class.eos_token_id
        if pad_id is None:
            pad_id = 0

        device = self.accelerator.device
        full_rows: List[List[int]] = []
        for prompt_ids, completion_ids in zip(prompt_ids_list, completion_ids_list):
            p = list(prompt_ids)
            if not p:
                bos_id = getattr(self.processing_class, "bos_token_id", None)
                p = [int(bos_id)] if bos_id is not None else [int(pad_id)]
            full_rows.append(p + list(completion_ids))

        max_len = max(1, max(len(x) for x in full_rows))
        input_ids = torch.full((len(full_rows), max_len), int(pad_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(full_rows), max_len), dtype=torch.long, device=device)
        prompt_lengths = torch.zeros(len(full_rows), dtype=torch.long, device=device)
        for i, row in enumerate(full_rows):
            row_t = torch.tensor(row, dtype=torch.long, device=device)
            input_ids[i, : row_t.numel()] = row_t
            attention_mask[i, : row_t.numel()] = 1
            prompt_lengths[i] = max(1, len(prompt_ids_list[i]))
        return input_ids, attention_mask, prompt_lengths

    def _build_loss_inputs(
        self,
        student_prompt_texts: Sequence[str],
        teacher_prompt_texts: Sequence[str],
        completion_ids_list: Sequence[Sequence[int]],
    ) -> Dict[str, torch.Tensor]:
        completion_ids_list = [list(x)[: self.max_completion_length] for x in completion_ids_list]
        max_comp_len = max(1, max((len(x) for x in completion_ids_list), default=1))
        device = self.accelerator.device

        student_prompt_ids = self._tokenize_prompts(
            student_prompt_texts,
            max_length=self.max_student_prompt_length,
        )
        teacher_prompt_ids = self._tokenize_prompts(
            teacher_prompt_texts,
            max_length=self.max_teacher_prompt_length,
        )

        student_input_ids, student_attention_mask, student_prompt_lengths = self._build_full_sequences(
            student_prompt_ids,
            completion_ids_list,
        )
        teacher_input_ids, teacher_attention_mask, teacher_prompt_lengths = self._build_full_sequences(
            teacher_prompt_ids,
            completion_ids_list,
        )

        labels = torch.full(
            (len(completion_ids_list), max_comp_len),
            -100,
            dtype=torch.long,
            device=device,
        )
        completion_mask = torch.zeros_like(labels, dtype=torch.float32)
        student_positions = torch.zeros_like(labels)
        teacher_positions = torch.zeros_like(labels)

        for i, completion_ids in enumerate(completion_ids_list):
            p_s = int(student_prompt_lengths[i].item())
            p_t = int(teacher_prompt_lengths[i].item())
            for j, token_id in enumerate(completion_ids):
                labels[i, j] = int(token_id)
                completion_mask[i, j] = 1.0
                student_positions[i, j] = max(0, p_s - 1 + j)
                teacher_positions[i, j] = max(0, p_t - 1 + j)

        return {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "student_positions": student_positions,
            "teacher_positions": teacher_positions,
            "labels": labels,
            "completion_mask": completion_mask,
        }

    def _model_forward(
        self,
        model,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ):
        """
        Qwen3/Transformers can return only the last N logits via ``logits_to_keep``.
        OPSD only needs completion-token logits, so this avoids materializing
        prompt-token vocab logits. Older model classes that do not support the
        kwarg fall back to the normal full-logits forward.
        """
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if logits_to_keep > 0:
            kwargs["logits_to_keep"] = int(logits_to_keep)
            try:
                return model(**kwargs)
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                kwargs.pop("logits_to_keep", None)
        return model(**kwargs)

    @staticmethod
    def _gather_token_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        positions = positions.clamp(min=0, max=max(0, logits.size(1) - 1))
        index = positions.unsqueeze(-1).expand(-1, -1, vocab_size)
        return torch.gather(logits, dim=1, index=index)

    def _forward_and_gather_token_logits(
        self,
        model,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        valid_positions = positions[labels != -100]
        if valid_positions.numel() == 0:
            logits_to_keep = 1
        else:
            min_position = int(valid_positions.min().item())
            logits_to_keep = int(input_ids.size(1)) - min_position
            logits_to_keep = max(1, min(int(input_ids.size(1)), logits_to_keep))

        outputs = self._model_forward(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )
        logits = outputs.logits
        window_start = int(input_ids.size(1)) - int(logits.size(1))
        relative_positions = positions - window_start
        gathered = self._gather_token_logits(logits, relative_positions)
        del outputs
        return gathered

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        completion_mask = inputs["completion_mask"]

        student_logits = self._forward_and_gather_token_logits(
            model,
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
            positions=inputs["student_positions"],
            labels=labels,
        )
        _empty_cache()

        unwrapped = self.accelerator.unwrap_model(model)
        if self.fixed_teacher and hasattr(unwrapped, "disable_adapter"):
            adapter_context = unwrapped.disable_adapter()
        else:
            adapter_context = nullcontext()

        with torch.no_grad(), adapter_context:
            teacher_logits = self._forward_and_gather_token_logits(
                model,
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                positions=inputs["teacher_positions"],
                labels=labels,
            )
        _empty_cache()

        loss = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=self.beta,
            temperature=self.temperature,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
        )
        loss = loss * self.lmbda

        with torch.no_grad():
            valid_tokens = completion_mask.sum().clamp(min=1.0)
            self._log_metric("opsd/loss_raw", float(loss.detach().float().item()))
            self._log_metric("opsd/lambda", self.lmbda)
            self._log_metric("opsd/beta", self.beta)
            self._log_metric("opsd/valid_tokens", float(valid_tokens.item()))
            lengths = completion_mask.sum(dim=1)
            self._log_metric("opsd/completion_length", self._reduce_scalar_mean(lengths.mean()))
            self._log_metric("opsd/completion_length_min", self._reduce_scalar_mean(lengths.min()))
            self._log_metric("opsd/completion_length_max", self._reduce_scalar_mean(lengths.max()))
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            student_probs = student_log_probs.exp()
            token_entropy = -(student_probs * student_log_probs).sum(dim=-1)
            entropy_mean = (token_entropy * completion_mask.float()).sum() / valid_tokens
            entropy_value = float(self.accelerator.gather_for_metrics(entropy_mean).mean().item())
            self._log_metric("entropy", entropy_value)
            self._log_metric("rl/entropy", entropy_value)
            self._log_metric("opsd/entropy", entropy_value)

        if return_outputs:
            return loss, {"loss": loss}
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._sync_vllm_if_needed()

        student_prompt_texts = inputs["student_prompt_text"]
        teacher_prompt_texts = inputs["teacher_prompt_text"]

        if self.use_vllm:
            completion_ids_list = self._generate_vllm(student_prompt_texts)
        else:
            if unwrap_model_for_generation is not None:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    completion_ids_list = self._generate_torch(unwrapped_model, student_prompt_texts)
            else:
                unwrapped_model = self.accelerator.unwrap_model(model)
                completion_ids_list = self._generate_torch(unwrapped_model, student_prompt_texts)

        loss_inputs = self._build_loss_inputs(
            student_prompt_texts=student_prompt_texts,
            teacher_prompt_texts=teacher_prompt_texts,
            completion_ids_list=completion_ids_list,
        )
        inputs.update(loss_inputs)
        self._record_generations(student_prompt_texts, teacher_prompt_texts, completion_ids_list)
        return super().training_step(model, inputs, num_items_in_batch)

    def _record_generations(
        self,
        student_prompt_texts: Sequence[str],
        teacher_prompt_texts: Sequence[str],
        completion_ids_list: Sequence[Sequence[int]],
    ) -> None:
        if self.save_generation_steps <= 0:
            return
        completion_texts = [
            self.processing_class.decode(list(ids), skip_special_tokens=False)
            for ids in completion_ids_list
        ]
        for prompt, teacher_prompt, completion in zip(
            student_prompt_texts,
            teacher_prompt_texts,
            completion_texts,
        ):
            self._generation_outputs_buffer.append(
                {
                    "step": int(getattr(self.state, "global_step", 0) or 0),
                    "prompt": prompt,
                    "teacher_prompt": teacher_prompt,
                    "completion": completion,
                }
            )
        step = int(getattr(self.state, "global_step", 0) or 0)
        if step <= 0 or step % self.save_generation_steps != 0:
            return
        if not self.accelerator.is_main_process:
            self._generation_outputs_buffer.clear()
            return
        out_dir = Path(self.args.output_dir) / "generations"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"generations_step_{step}.json"
        payload = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self._generation_outputs_buffer.clear()

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(values) / max(1, len(values))
            for key, values in self._metrics.get(mode, {}).items()
            if values
        }
        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        logs = {**logs, **metrics}
        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)
        self._metrics.get(mode, {}).clear()
