from pathlib import Path

import pytest

from beatforge import LocalServerSpec, launch_local_server


def test_local_server_command_is_loopback_and_shell_free(tmp_path):
    binary = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")
    spec = LocalServerSpec(binary, model, port=8123, context_size=4096, threads=4)
    assert spec.validate() == []
    assert spec.command() == [
        str(binary.resolve()), "-m", str(model.resolve()),
        "--host", "127.0.0.1", "--port", "8123",
        "--ctx-size", "4096", "--alias", "beatforge-local", "-t", "4",
    ]

    calls = {}

    class Process:
        pid = 1234

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return Process()

    process = launch_local_server(spec, popen=fake_popen)
    assert process.pid == 1234
    assert calls["kwargs"]["shell"] is False
    assert calls["kwargs"]["cwd"] == str(binary.resolve().parent)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.test"])
def test_local_server_rejects_non_loopback_bind(tmp_path, host):
    binary = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="loopback"):
        LocalServerSpec(binary, model, host=host).command()


def test_local_server_rejects_missing_inputs(tmp_path):
    with pytest.raises(ValueError, match="binary does not exist"):
        LocalServerSpec(tmp_path / "missing.exe", tmp_path / "model.gguf").command()


def test_local_server_command_includes_validated_mmproj(tmp_path):
    binary = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projector")

    spec = LocalServerSpec(binary, model, mmproj=mmproj)
    assert spec.validate() == []
    command = spec.command()
    index = command.index("--mmproj")
    assert command[index + 1] == str(mmproj.resolve())

    with pytest.raises(ValueError, match="projector file does not exist"):
        LocalServerSpec(binary, model, mmproj=tmp_path / "missing-mmproj.gguf").command()
