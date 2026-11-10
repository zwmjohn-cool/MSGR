"""Atomic training log recovery helpers."""
import json
import os

def _jsonl(path):
    """Tolerate only a partially written final record, never internal damage."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
    return rows

def _atomic(path, text):
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(text, encoding='utf8')
    temp.replace(path)
