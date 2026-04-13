import os
from pathlib import Path
from typing import Iterable


def load_env_files(paths: Iterable[str], override: bool = False) -> None:
    for path in paths:
        load_env_file(path, override=override)


def load_env_file(path: str, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue

        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export ") :].strip()

    key, separator, value = line.partition("=")
    key = key.strip()
    if not separator or not key:
        return None

    return key, _clean_env_value(value.strip())


def _clean_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()

    return value
