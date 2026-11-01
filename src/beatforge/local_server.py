"""Safe launcher configuration for a local llama.cpp OpenAI-compatible server."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class LocalServerSpec:
    binary: str | Path
    model: str | Path
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 8192
    threads: int | None = None
    mmproj: str | Path | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        binary = Path(self.binary)
        model = Path(self.model)
        if not binary.is_file():
            errors.append(f"llama server binary does not exist: {binary}")
        if not model.is_file():
            errors.append(f"model file does not exist: {model}")
        if self.mmproj is not None and not Path(self.mmproj).is_file():
            errors.append(f"multimodal projector file does not exist: {self.mmproj}")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            errors.append("local server host must be loopback")
        if not 1 <= self.port <= 65535:
            errors.append("local server port must be between 1 and 65535")
        if self.context_size <= 0:
            errors.append("context_size must be positive")
        if self.threads is not None and self.threads <= 0:
            errors.append("threads must be positive")
        return errors

    def command(self) -> list[str]:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))
        command = [str(Path(self.binary).resolve()), "-m", str(Path(self.model).resolve()),
                   "--host", self.host, "--port", str(self.port),
                   "--ctx-size", str(self.context_size), "--alias", "beatforge-local"]
        if self.mmproj is not None:
            command.extend(["--mmproj", str(Path(self.mmproj).resolve())])
        if self.threads is not None:
            command.extend(["-t", str(self.threads)])
        return command


def launch_local_server(spec: LocalServerSpec, *, popen: Callable[..., Any] = subprocess.Popen) -> Any:
    """Start llama.cpp without a shell or a non-loopback bind."""
    return popen(spec.command(), cwd=str(Path(spec.binary).resolve().parent), shell=False)

