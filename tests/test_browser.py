from pathlib import Path

import pytest

from beatforge import BrowserSessionSpec, launch_browser


def test_browser_spec_is_allowlisted_and_builds_extension_args(tmp_path):
    extension = tmp_path / "anti-debug"
    extension.mkdir()
    spec = BrowserSessionSpec("chrome", str(tmp_path / "profile"), str(extension),
                              start_urls=("https://polyhaven.com/",))
    assert spec.validate() == []
    args = spec.command_args()
    assert any(arg.startswith("--load-extension=") for arg in args)


def test_browser_spec_rejects_external_domains_and_editor_scripts(tmp_path):
    spec = BrowserSessionSpec("chrome", str(tmp_path / "profile"),
                              start_urls=("https://example.com/",), allow_editor_scripts=True)
    errors = spec.validate()
    assert any("domain allowlist" in error for error in errors)
    assert any("editor scripts" in error for error in errors)
    with pytest.raises(ValueError):
        spec.command_args()


def test_launch_browser_creates_profile_and_returns_process_handle(tmp_path):
    calls = []

    def fake_popen(args, cwd):
        calls.append((args, cwd))
        return "process"

    spec = BrowserSessionSpec("chrome", str(tmp_path / "profile"),
                              start_urls=("https://polyhaven.com/",))
    result = launch_browser(spec, popen=fake_popen)
    assert result == "process"
    assert (tmp_path / "profile").is_dir()
    assert calls and calls[0][0][0] == "chrome"
    assert calls[0][1] == str(tmp_path / "profile")
