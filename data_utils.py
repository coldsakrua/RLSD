from pathlib import Path
from typing import Iterable, Optional

from datasets import Dataset, load_dataset


_PROMPT_KEYS = ("prompt", "problem", "question", "query", "input", "instruction")
_SOLUTION_KEYS = ("solution", "answer", "ground_truth", "target", "reference")
_AGGREGATED_L3PLUS_KEYS = ("problem", "solution", "level", "type", "subject")


def _pick_key(candidates: Iterable[str], columns: Iterable[str]) -> Optional[str]:
    col_set = set(columns)
    for key in candidates:
        if key in col_set:
            return key
    return None


def _resolve_data_file(dataset_path: str, split: str) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path

    candidates = [
        path / f"{split}.jsonl",
        path / f"{split}.json",
        path / f"{split}.parquet",
        path / "data.jsonl",
        path / "data.json",
        path / "data.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find dataset file under {dataset_path}. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def _load_single_file(path: Path, split: str) -> Dataset:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        return load_dataset("json", data_files={split: str(path)}, split=split)
    if suffix == ".parquet":
        return load_dataset("parquet", data_files={split: str(path)}, split=split)
    raise ValueError(f"Unsupported dataset format: {path}")


def _looks_like_aggregated_l3plus(columns: Iterable[str]) -> bool:
    col_set = set(columns)
    return all(k in col_set for k in _AGGREGATED_L3PLUS_KEYS)


def _non_empty_text(x) -> bool:
    if x is None:
        return False
    return bool(str(x).strip())


def load_rlsd_dataset(dataset_path: str, split: str = "train") -> Dataset:
    data_file = _resolve_data_file(dataset_path, split)
    ds = _load_single_file(data_file, split=split)

    if _looks_like_aggregated_l3plus(ds.column_names):
        def _normalize_l3plus(row):
            row["prompt"] = str(row["problem"]).strip()
            row["solution"] = str(row["solution"]).strip()
            row["problem_level"] = str(row.get("level", "")).strip()
            row["problem_type"] = str(row.get("type", "")).strip()
            row["problem_subject"] = str(row.get("subject", "")).strip()
            return row

        ds = ds.map(_normalize_l3plus, desc="Normalizing aggregated_l3plus schema")
        ds = ds.filter(
            lambda row: _non_empty_text(row["prompt"]) and _non_empty_text(row["solution"]),
            desc="Filtering empty prompt/solution rows",
        )
        return ds

    prompt_key = _pick_key(_PROMPT_KEYS, ds.column_names)
    solution_key = _pick_key(_SOLUTION_KEYS, ds.column_names)
    if prompt_key is None or solution_key is None:
        raise ValueError(
            f"Failed to infer prompt/solution columns from {ds.column_names}. "
            f"Expected prompt in {_PROMPT_KEYS}, solution in {_SOLUTION_KEYS}."
        )

    def _normalize(row):
        prompt = row[prompt_key]
        solution = row[solution_key]
        row["prompt"] = prompt if isinstance(prompt, str) else str(prompt)
        row["solution"] = solution if isinstance(solution, str) else str(solution)
        return row

    ds = ds.map(_normalize, desc="Normalizing prompt and solution columns")
    return ds
