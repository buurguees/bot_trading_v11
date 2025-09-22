import os, json, pickle
from typing import Any

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_pickle(obj: Any, path: str):
    ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(obj: Any, path: str, indent: int = 2):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
