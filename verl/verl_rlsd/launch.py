from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any

from . import advantage as custom_advantage
from .qwen3_chat_template import (
    install_qwen3_no_think_chat_template,
    strip_empty_thinking_enabled,
    strip_empty_thinking_generation_prompt,
)
from .rollout_dump import patch_rollout_dump
from .rollout_metrics import patch_rollout_length_logging
from .teacher_ema import patch_teacher_ema


def _cfg_get(config: Any, path: str, default: Any = None) -> Any:
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _patch_compute_advantage() -> None:
    try:
        ray_trainer = importlib.import_module("verl.trainer.ppo.ray_trainer")
    except Exception:
        return
    original = getattr(ray_trainer, "compute_advantage", None)
    if original is None:
        return

    def patched_compute_advantage(data: Any, adv_estimator: Any, gamma: float = 1.0, lam: float = 1.0, num_repeat: int = 1, config: Any = None, **kwargs: Any):
        name = getattr(adv_estimator, "value", adv_estimator)
        custom_name = _cfg_get(config, "algorithm.rlsd.custom_adv_estimator", None)
        if custom_name:
            name = str(custom_name)
        if str(name) in custom_advantage.CUSTOM_ADV_ESTIMATORS:
            return custom_advantage.compute_custom_advantage(data, str(name), config)
        return original(
            data=data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            config=config,
            **kwargs,
        )

    ray_trainer.compute_advantage = patched_compute_advantage


_RAY_INIT_OLD = "ray.init(namespace=namespace)"
_RAY_INIT_NEW = (
    'ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), '
    "namespace=namespace, ignore_reinit_error=True)"
)
_RLSD_STRIP_EMPTY_THINKING_PATCH = "# RLSD_STRIP_EMPTY_THINKING_PATCH"
_RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH = "# RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH"
_RLSD_VLLM_CHAT_TEMPLATE_PATCH = "# RLSD_VLLM_CHAT_TEMPLATE_PATCH"
_APPLY_CHAT_NO_THINK_OLD = """    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            return _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            return _orig_apply_chat(messages, *args, **kw)"""
_APPLY_CHAT_NO_THINK_NEW = """    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            out = _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            out = _orig_apply_chat(messages, *args, **kw)
        if (
            isinstance(out, str)
            and _rlsd_os.environ.get("STRIP_EMPTY_THINKING_GENERATION_PROMPT", "false").strip().lower()
            in {"1", "true", "yes", "on"}
            and kw.get("add_generation_prompt", True)
        ):
            try:
                from verl_rlsd.qwen3_chat_template import strip_empty_thinking_generation_prompt
                out = strip_empty_thinking_generation_prompt(out)
            except Exception:
                pass
        return out"""


_ASYNC_VLLM_ENGINE_KWARGS_MARKER = "# RLSD_ASYNC_VLLM_ENGINE_KWARGS_PATCH"
_ASYNC_VLLM_ENGINE_KWARGS_OLD = """        print(f"override_generation_config: {kwargs}")

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor if os.environ.get("VERL_VLLM_USE_RAY_BACKEND", "1") == "1" else None,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
        )"""
_ASYNC_VLLM_ENGINE_KWARGS_NEW = """        print(f"override_generation_config: {kwargs}")

        from copy import deepcopy

        from omegaconf import OmegaConf

        engine_kwargs = {}
        if getattr(config, "engine_kwargs", None) is not None and getattr(
            config.engine_kwargs, "vllm", None
        ) is not None:
            engine_kwargs = OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        model_config = self.config.model
        lora_rank = int(getattr(model_config, "lora_rank", 0) or 0)
        if lora_rank > 0 and not engine_kwargs.get("enable_lora"):
            engine_kwargs["enable_lora"] = True
            engine_kwargs.setdefault("max_lora_rank", lora_rank)
            engine_kwargs.setdefault("max_loras", max(int(engine_kwargs.get("max_loras", 0) or 0), 1))
        """ + _ASYNC_VLLM_ENGINE_KWARGS_MARKER + """

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor if os.environ.get("VERL_VLLM_USE_RAY_BACKEND", "1") == "1" else None,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
            **engine_kwargs,
        )"""


def _patch_verl_vllm_async_server_on_disk() -> None:
    """Patch installed verl async vLLM server for Ray cluster + LoRA engine kwargs."""
    try:
        mod = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    except Exception:
        return
    path = getattr(mod, "__file__", None)
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    changed = False
    if _RAY_INIT_OLD in text and _RAY_INIT_NEW not in text:
        text = text.replace(_RAY_INIT_OLD, _RAY_INIT_NEW, 1)
        changed = True
    if _ASYNC_VLLM_ENGINE_KWARGS_MARKER not in text:
        if _ASYNC_VLLM_ENGINE_KWARGS_OLD not in text:
            return
        text = text.replace(_ASYNC_VLLM_ENGINE_KWARGS_OLD, _ASYNC_VLLM_ENGINE_KWARGS_NEW, 1)
        changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


_FSDP_VLLM_ADD_LORA_OLD = "self.inference_engine.llm_engine.add_lora(lora_reqest)"
_FSDP_VLLM_ADD_LORA_CALL = (
    '__import__("verl_rlsd.launch", fromlist=["_rlsd_add_lora"]).'
    '_rlsd_add_lora(self, lora_reqest)  # RLSD_ASYNC_VLLM_ADD_LORA_PATCH'
)


def _rlsd_add_lora(sharding_manager: Any, lora_request: Any) -> None:
    """Compatibility wrapper for veRL LoRA sync across vLLM v0/v1 objects."""
    import asyncio
    import inspect

    inference_engine = getattr(sharding_manager, "inference_engine", None)
    candidates = [
        getattr(inference_engine, "llm_engine", None),
        inference_engine,
        getattr(inference_engine, "worker", None),
        getattr(getattr(inference_engine, "worker", None), "model_runner", None),
        getattr(sharding_manager, "model_runner", None),
    ]
    seen: set[int] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        add_lora = getattr(candidate, "add_lora", None)
        if add_lora is None:
            continue
        try:
            result = add_lora(lora_request)
        except AttributeError as exc:
            if "lora_manager" in str(exc):
                last_error = exc
                continue
            raise
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is None:
                asyncio.run(result)
            elif loop.is_running():
                asyncio.run_coroutine_threadsafe(result, loop).result()
            else:
                loop.run_until_complete(result)
        return
    if last_error is not None:
        raise AttributeError(
            "vLLM exposes add_lora but was started without a LoRA manager. "
            "Enable actor_rollout_ref.rollout.engine_kwargs.vllm.enable_lora=true "
            "and set max_loras/max_lora_rank for LoRA rollout sync. If LoRA "
            "merge is enabled, also set actor_rollout_ref.model.lora.merge=false "
            "so vLLM starts in adapter mode."
        ) from last_error
    raise AttributeError(
        "RLSD failed to find a vLLM add_lora API; this veRL/vLLM combination "
        "does not expose llm_engine.add_lora or an equivalent add_lora method."
    )


def _find_module_file(module_name: str, relative_path: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
        path = getattr(spec, "origin", None) if spec is not None else None
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    for base in sys.path:
        if not base:
            continue
        path = os.path.join(base, relative_path)
        if os.path.isfile(path):
            return path
    return None


def _repair_bad_add_lora_patch(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "RLSD_ASYNC_VLLM_ADD_LORA_PATCH" in line and "_rlsd_add_lora" not in line:
            start = i
            end = None
            limit = min(len(lines), i + 80)
            j = i + 1
            while j < limit:
                if "_rlsd_loop.run_until_complete(_rlsd_lora_result)" in lines[j]:
                    end = j + 1
                    break
                j += 1
            if end is None:
                out.append(line)
                i = start + 1
                continue
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}{_FSDP_VLLM_ADD_LORA_CALL}{newline}")
            changed = True
            i = end
            continue
        out.append(line)
        i += 1
    return "".join(out), changed


def _patch_fsdp_vllm_add_lora_on_disk() -> None:
    """Patch old veRL FSDP-vLLM sharding to support vLLM v1 LoRA sync."""
    path = _find_module_file(
        "verl.workers.sharding_manager.fsdp_vllm",
        os.path.join("verl", "workers", "sharding_manager", "fsdp_vllm.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text, repaired = _repair_bad_add_lora_patch(text)
    if not repaired and "RLSD_ASYNC_VLLM_ADD_LORA_PATCH" in text:
        return
    if not repaired and _FSDP_VLLM_ADD_LORA_OLD not in text:
        return
    if not repaired:
        text = text.replace(_FSDP_VLLM_ADD_LORA_OLD, _FSDP_VLLM_ADD_LORA_CALL, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


_RLSD_APPLY_CHAT_TEMPLATE_KWARGS_MARKER = "# RLSD_APPLY_CHAT_TEMPLATE_KWARGS"
_RLSD_DISABLE_THINKING_TOKENIZER_MARKER = "# RLSD_DISABLE_THINKING_IN_CHAT_TEMPLATE_PATCH"
_RLSD_CHAT_SCHEDULER_DISABLE_THINKING_MARKER = "# RLSD_CHAT_SCHEDULER_DISABLE_THINKING_PATCH"


def _rlsd_disable_thinking_enabled() -> bool:
    raw = os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "")
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _patch_rl_dataset_apply_chat_template_kwargs_on_disk() -> None:
    """Pass data.apply_chat_template_kwargs (e.g. enable_thinking=False) in Ray workers."""
    path = _find_module_file(
        "verl.utils.dataset.rl_dataset",
        os.path.join("verl", "utils", "dataset", "rl_dataset.py"),
    )
    if not path or _RLSD_APPLY_CHAT_TEMPLATE_KWARGS_MARKER in open(path, encoding="utf-8").read():
        return

    with open(path, encoding="utf-8") as f:
        text = f.read()

    init_old = '        self.chat_template_func = config.get("chat_template_func", None)\n        self.need_tools_kwargs = config.get("need_tools_kwargs", False)'
    init_new = (
        '        self.chat_template_func = config.get("chat_template_func", None)\n'
        "        self.apply_chat_template_kwargs = dict(config.get(\"apply_chat_template_kwargs\") or {})  "
        + _RLSD_APPLY_CHAT_TEMPLATE_KWARGS_MARKER
        + "\n        self.need_tools_kwargs = config.get(\"need_tools_kwargs\", False)"
    )
    if init_old not in text:
        return
    text = text.replace(init_old, init_new, 1)

    replacements = [
        (
            "self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)",
            "self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)",
        ),
        (
            "return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))",
            "return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs))",
        ),
        (
            "self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)",
            "self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            return
        text = text.replace(old, new)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_chat_scheduler_disable_thinking_on_disk() -> None:
    """Pass chat_template_kwargs to vLLM async chat/completions API (rollout.mode=async)."""
    if not _rlsd_disable_thinking_enabled():
        return
    path = _find_module_file(
        "verl.workers.rollout.chat_scheduler",
        os.path.join("verl", "workers", "rollout", "chat_scheduler.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_CHAT_SCHEDULER_DISABLE_THINKING_MARKER in text:
        return

    helper = '''
def _rlsd_chat_template_kwargs() -> dict:
    """Qwen3 non-thinking mode for async vLLM chat/completions requests."""
    import os as _rlsd_os

    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }:
        return {"enable_thinking": False}
    return {}


''' + _RLSD_CHAT_SCHEDULER_DISABLE_THINKING_MARKER + "\n"

    anchor = "logger = logging.getLogger(__file__)"
    if anchor not in text:
        return
    text = text.replace(anchor, anchor + "\n\n" + helper, 1)

    extra_body_old = '''    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None'''
    extra_body_new = '''    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        chat_kwargs = _rlsd_chat_template_kwargs()
        if chat_kwargs:
            return {"chat_template_kwargs": chat_kwargs}
        return None'''
    if extra_body_old not in text:
        return
    text = text.replace(extra_body_old, extra_body_new, 1)

    postprocess_old = (
        "        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, "
        "add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch[\"raw_prompt\"]]"
    )
    postprocess_new = (
        "        _rlsd_ctkw = _rlsd_chat_template_kwargs()\n"
        "        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, "
        "add_generation_prompt=True, tokenize=False, **_rlsd_ctkw) "
        "for prompt in batch.non_tensor_batch[\"raw_prompt\"]]"
    )
    if postprocess_old not in text:
        return
    text = text.replace(postprocess_old, postprocess_new, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_hf_tokenizer_disable_thinking_on_disk() -> None:
    """Apply enable_thinking=False inside hf_tokenizer/hf_processor for all Ray worker processes."""
    if not _rlsd_disable_thinking_enabled():
        return
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_DISABLE_THINKING_TOKENIZER_MARKER in text:
        return

    helper = '''
def _rlsd_wrap_disable_thinking(tokenizer):
    """Force enable_thinking=False for Qwen3 chat templates in every worker process."""
    import os as _rlsd_os

    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    if getattr(tokenizer.apply_chat_template, "_rlsd_disable_thinking", False):
        return tokenizer
    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() not in {
        "1", "true", "yes", "y", "on",
    }:
        return tokenizer

    try:
        from verl_rlsd.qwen3_chat_template import install_qwen3_no_think_chat_template
        install_qwen3_no_think_chat_template(tokenizer)
    except Exception:
        pass

    _orig_apply_chat = tokenizer.apply_chat_template

    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            out = _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            out = _orig_apply_chat(messages, *args, **kw)
        if (
            isinstance(out, str)
            and _rlsd_os.environ.get("STRIP_EMPTY_THINKING_GENERATION_PROMPT", "false").strip().lower()
            in {"1", "true", "yes", "on"}
            and kw.get("add_generation_prompt", True)
        ):
            try:
                from verl_rlsd.qwen3_chat_template import strip_empty_thinking_generation_prompt
                out = strip_empty_thinking_generation_prompt(out)
            except Exception:
                pass
        return out

    _apply_chat_no_think._rlsd_disable_thinking = True
    tokenizer.apply_chat_template = _apply_chat_no_think
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and inner is not tokenizer and hasattr(inner, "apply_chat_template"):
        _rlsd_wrap_disable_thinking(inner)
    return tokenizer

''' + _RLSD_DISABLE_THINKING_TOKENIZER_MARKER + "\n"

    anchor = "def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):"
    if anchor not in text:
        return
    text = text.replace(anchor, helper + anchor, 1)

    old_return = """    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor"""
    new_return = """    if correct_pad_token:
        set_pad_token_id(tokenizer)
    tokenizer = _rlsd_wrap_disable_thinking(tokenizer)
    return tokenizer


def hf_processor"""
    if old_return not in text:
        return
    text = text.replace(old_return, new_return, 1)

    old_proc_return = """    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor"""
    new_proc_return = """    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    processor = _rlsd_wrap_disable_thinking(processor)
    return processor"""
    if old_proc_return not in text:
        return
    text = text.replace(old_proc_return, new_proc_return, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_hf_tokenizer_strip_empty_thinking_on_disk() -> None:
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path or _RLSD_STRIP_EMPTY_THINKING_PATCH in open(path, encoding="utf-8").read():
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _APPLY_CHAT_NO_THINK_OLD not in text:
        return
    text = text.replace(_APPLY_CHAT_NO_THINK_OLD, _APPLY_CHAT_NO_THINK_NEW, 1)
    text = text.replace(
        _RLSD_DISABLE_THINKING_TOKENIZER_MARKER,
        _RLSD_STRIP_EMPTY_THINKING_PATCH + "\n" + _RLSD_DISABLE_THINKING_TOKENIZER_MARKER,
        1,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_hf_tokenizer_install_no_think_template_on_disk() -> None:
    """Upgrade Ray-worker tokenizer patch to install Qwen3 no-think chat template."""
    if not _rlsd_disable_thinking_enabled():
        return
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH in text:
        return
    anchor = """    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() not in {
        "1", "true", "yes", "y", "on",
    }:
        return tokenizer

    _orig_apply_chat = tokenizer.apply_chat_template"""
    insert = """    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() not in {
        "1", "true", "yes", "y", "on",
    }:
        return tokenizer

    try:
        from verl_rlsd.qwen3_chat_template import install_qwen3_no_think_chat_template
        install_qwen3_no_think_chat_template(tokenizer)
    except Exception:
        pass
""" + _RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH + """

    _orig_apply_chat = tokenizer.apply_chat_template"""
    if anchor not in text:
        return
    text = text.replace(anchor, insert, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_vllm_openai_chat_template_on_disk() -> None:
    if not strip_empty_thinking_enabled():
        return
    path = _find_module_file(
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        os.path.join("verl", "workers", "rollout", "vllm_rollout", "vllm_async_server.py"),
    )
    if not path or _RLSD_VLLM_CHAT_TEMPLATE_PATCH in open(path, encoding="utf-8").read():
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    anchor = "        # build serving chat\n        model_config = self.engine.model_config"
    if anchor not in text:
        return
    insert = (
        "        # build serving chat\n"
        "        _rlsd_chat_template = None\n"
        "        if os.environ.get(\"STRIP_EMPTY_THINKING_GENERATION_PROMPT\", \"false\").strip().lower() in {\n"
        "            \"1\", \"true\", \"yes\", \"on\",\n"
        "        }:\n"
        "            try:\n"
        "                from verl_rlsd.qwen3_chat_template import load_qwen3_chat_template_without_empty_thinking\n"
        "                _rlsd_chat_template = load_qwen3_chat_template_without_empty_thinking(local_path)\n"
        "            except Exception as _rlsd_ct_exc:\n"
        "                print(f\"[chat_template] failed to load stripped Qwen3 template: {_rlsd_ct_exc}\", flush=True)\n"
        "        model_config = self.engine.model_config"
        + _RLSD_VLLM_CHAT_TEMPLATE_PATCH
        + "\n"
    )
    text = text.replace(anchor, insert, 1)
    text = text.replace("            chat_template=None,", "            chat_template=_rlsd_chat_template,", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_ray_init_for_vllm() -> None:
    """Ensure Ray workers inherit the vLLM mode and training cluster RAY_ADDRESS."""
    os.environ.setdefault("VLLM_USE_V1", "1")
    try:
        import ray
    except Exception:
        return
    if getattr(ray.init, "_rlsd_vllm_env_patched", False):
        return
    original_init = ray.init

    def patched_init(*args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars.setdefault("VLLM_USE_V1", os.environ.get("VLLM_USE_V1", "1"))
        if os.environ.get("RLSD_MERGE_LORA_FOR_ASYNC_VLLM"):
            env_vars.setdefault(
                "RLSD_MERGE_LORA_FOR_ASYNC_VLLM",
                os.environ["RLSD_MERGE_LORA_FOR_ASYNC_VLLM"],
            )
        if os.environ.get("RAY_ADDRESS"):
            env_vars.setdefault("RAY_ADDRESS", os.environ["RAY_ADDRESS"])
        if os.environ.get("RAY_TMPDIR"):
            env_vars.setdefault("RAY_TMPDIR", os.environ["RAY_TMPDIR"])
        if os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE"):
            env_vars.setdefault(
                "DISABLE_THINKING_IN_CHAT_TEMPLATE",
                os.environ["DISABLE_THINKING_IN_CHAT_TEMPLATE"],
            )
        if os.environ.get("STRIP_EMPTY_THINKING_GENERATION_PROMPT"):
            env_vars.setdefault(
                "STRIP_EMPTY_THINKING_GENERATION_PROMPT",
                os.environ["STRIP_EMPTY_THINKING_GENERATION_PROMPT"],
            )
        if os.environ.get("MATH_PROMPT_PREFIX"):
            env_vars.setdefault("MATH_PROMPT_PREFIX", os.environ["MATH_PROMPT_PREFIX"])
        if os.environ.get("STRIP_DAPO_PROMPT_BOILERPLATE"):
            env_vars.setdefault(
                "STRIP_DAPO_PROMPT_BOILERPLATE",
                os.environ["STRIP_DAPO_PROMPT_BOILERPLATE"],
            )
        runtime_env["env_vars"] = env_vars
        kwargs["runtime_env"] = runtime_env
        result = original_init(*args, **kwargs)
        try:
            gcs = ray.get_runtime_context().gcs_address
            if gcs:
                os.environ["RAY_ADDRESS"] = gcs
        except Exception:
            pass
        return result

    patched_init._rlsd_vllm_env_patched = True
    ray.init = patched_init


def _maybe_strip_empty_thinking_prompt(result: Any, kwargs: dict[str, Any]) -> Any:
    if not isinstance(result, str) or not strip_empty_thinking_enabled():
        return result
    add_generation_prompt = kwargs.get("add_generation_prompt")
    if add_generation_prompt is None:
        add_generation_prompt = True
    if add_generation_prompt:
        return strip_empty_thinking_generation_prompt(result)
    return result


def _disable_thinking_on_tokenizer(tokenizer) -> None:
    """Same monkey-patch as opsd_train_anchor_strict_split_flip_wrong_boost.py."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return
    if getattr(tokenizer.apply_chat_template, "_rlsd_disable_thinking", False):
        return

    install_qwen3_no_think_chat_template(tokenizer)

    _orig_apply_chat = tokenizer.apply_chat_template

    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            out = _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            out = _orig_apply_chat(messages, *args, **kw)
        return _maybe_strip_empty_thinking_prompt(out, kw)

    _apply_chat_no_think._rlsd_disable_thinking = True
    tokenizer.apply_chat_template = _apply_chat_no_think


def _patch_disable_thinking_in_chat_template() -> None:
    if not _rlsd_disable_thinking_enabled():
        return
    try:
        tokenizer_mod = importlib.import_module("verl.utils.tokenizer")
    except Exception:
        return

    original_hf_tokenizer = tokenizer_mod.hf_tokenizer
    original_hf_processor = tokenizer_mod.hf_processor
    if getattr(original_hf_tokenizer, "_rlsd_disable_thinking", False):
        return

    def patched_hf_tokenizer(*args, **kwargs):
        tokenizer = original_hf_tokenizer(*args, **kwargs)
        _disable_thinking_on_tokenizer(tokenizer)
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None and inner is not tokenizer:
            _disable_thinking_on_tokenizer(inner)
        return tokenizer

    def patched_hf_processor(*args, **kwargs):
        processor = original_hf_processor(*args, **kwargs)
        _disable_thinking_on_tokenizer(processor)
        return processor

    patched_hf_tokenizer._rlsd_disable_thinking = True
    patched_hf_processor._rlsd_disable_thinking = True
    tokenizer_mod.hf_tokenizer = patched_hf_tokenizer
    tokenizer_mod.hf_processor = patched_hf_processor
    try:
        utils_mod = importlib.import_module("verl.utils")
        utils_mod.hf_tokenizer = patched_hf_tokenizer
        utils_mod.hf_processor = patched_hf_processor
    except Exception:
        pass
    print("[chat_template] enable_thinking=False (disable_thinking_in_chat_template=True)", flush=True)


def main() -> None:
    _patch_hf_tokenizer_disable_thinking_on_disk()
    _patch_hf_tokenizer_strip_empty_thinking_on_disk()
    _patch_hf_tokenizer_install_no_think_template_on_disk()
    _patch_rl_dataset_apply_chat_template_kwargs_on_disk()
    _patch_chat_scheduler_disable_thinking_on_disk()
    _patch_disable_thinking_in_chat_template()
    _patch_verl_vllm_async_server_on_disk()
    _patch_vllm_openai_chat_template_on_disk()
    _patch_fsdp_vllm_add_lora_on_disk()
    _patch_ray_init_for_vllm()
    _patch_compute_advantage()
    patch_rollout_length_logging()
    patch_rollout_dump()
    patch_teacher_ema()
    module_name = os.environ.get("VERL_MAIN_MODULE", "verl.trainer.main_ppo")
    module = importlib.import_module(module_name)
    entry = getattr(module, "main", None)
    if entry is None:
        raise RuntimeError(f"{module_name} does not expose a main() function.")
    sys.exit(entry())


if __name__ == "__main__":
    main()
