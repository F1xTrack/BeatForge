"""Declarative browser session configuration for asset discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import os
import shutil
import subprocess
from typing import Any, Callable
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class BrowserSessionSpec:
    browser_executable: str | None
    profile_dir: str
    anti_debug_extension_dir: str | None = None
    allowed_domains: tuple[str, ...] = ("polyhaven.com", "ambientcg.com", "dl.polyhaven.org")
    start_urls: tuple[str, ...] = ()
    download_dir: str | None = None
    allow_editor_scripts: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.profile_dir:
            errors.append("profile_dir is required")
        if self.anti_debug_extension_dir and not Path(self.anti_debug_extension_dir).is_dir():
            errors.append("anti_debug_extension_dir does not exist")
        if self.allow_editor_scripts:
            errors.append("automatic editor scripts remain disabled by policy")
        for url in self.start_urls:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                errors.append(f"unsupported browser URL scheme: {url}")
            if parsed.hostname and not any(parsed.hostname == domain or parsed.hostname.endswith("." + domain)
                                           for domain in self.allowed_domains):
                errors.append(f"URL is outside the asset domain allowlist: {url}")
        return errors

    def command_args(self) -> list[str]:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))
        if not self.browser_executable:
            raise ValueError("browser_executable is required for direct process launch")
        args = [self.browser_executable, f"--user-data-dir={self.profile_dir}", "--new-window"]
        if self.download_dir:
            args.append(f"--download.default_directory={self.download_dir}")
        if self.anti_debug_extension_dir:
            args.extend([f"--disable-extensions-except={self.anti_debug_extension_dir}",
                         f"--load-extension={self.anti_debug_extension_dir}"])
        args.extend(self.start_urls)
        return args


def launch_browser(spec: BrowserSessionSpec, *, popen: Callable[..., Any] = subprocess.Popen) -> Any:
    """Launch the isolated asset-search browser after validation.

    The caller owns the returned process handle. No downloaded file is opened
    and no editor script is enabled by this helper.
    """
    errors = spec.validate()
    if errors:
        raise ValueError("; ".join(errors))
    profile = Path(spec.profile_dir)
    profile.mkdir(parents=True, exist_ok=True)
    if spec.download_dir:
        Path(spec.download_dir).mkdir(parents=True, exist_ok=True)
    return popen(spec.command_args(), cwd=str(profile))


def discover_chrome_executable() -> str:
    """Find an installed Chrome executable without starting it."""
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Google/Chrome/Application/chrome.exe",
    ]
    found = next((path for path in candidates if path.is_file()), None)
    if found:
        return str(found)
    command = shutil.which("chrome") or shutil.which("chrome.exe")
    if command:
        return command
    raise FileNotFoundError("Google Chrome was not found; set --browser-executable")


def discover_anti_debug_extension(chrome_user_data: str | Path | None = None) -> str:
    """Find the installed Anti Anti Debug extension by its Chrome extension ID."""
    root = Path(chrome_user_data) if chrome_user_data else (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data")
    extension_root = root / "Default/Extensions/mnmnmcmdkigakhlfkcdimghndnmomfeo"
    if extension_root.is_dir():
        versions = sorted((path for path in extension_root.iterdir() if path.is_dir()),
                          key=lambda path: path.name, reverse=True)
        for version in versions:
            try:
                manifest = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("name") == "Anti Anti Debug":
                return str(version)
    raise FileNotFoundError("Anti Anti Debug extension was not found; set --anti-debug-extension")


class PlaywrightBrowserSession:
    """Visible isolated Chrome session with the requested extension loaded."""

    def __init__(self, spec: BrowserSessionSpec):
        self.spec = spec
        self._playwright = None
        self.context = None
        self.page = None

    def __enter__(self) -> "PlaywrightBrowserSession":
        errors = self.spec.validate()
        if errors:
            raise ValueError("; ".join(errors))
        if not self.spec.anti_debug_extension_dir:
            raise ValueError("an Anti Anti Debug extension directory is required")
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "browser asset search requires the optional 'browser' extra (playwright)"
            ) from exc
        profile = Path(self.spec.profile_dir).resolve()
        profile.mkdir(parents=True, exist_ok=True)
        if self.spec.download_dir:
            Path(self.spec.download_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        args = [f"--disable-extensions-except={self.spec.anti_debug_extension_dir}",
                f"--load-extension={self.spec.anti_debug_extension_dir}"]
        try:
            self.context = self._playwright.chromium.launch_persistent_context(
                str(profile), channel="chromium", headless=False, args=args,
                downloads_path=self.spec.download_dir,
            )
            self.context.set_extra_http_headers({"User-Agent": "BeatForge/0.1 asset discovery"})
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            for url in self.spec.start_urls:
                self._goto_allowed(url)
            return self
        except Exception:
            self.close()
            raise

    def _goto_allowed(self, url: str) -> Any:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not any(
            host == domain or host.endswith("." + domain)
            for domain in self.spec.allowed_domains
        ):
            raise ValueError("browser navigation is outside the HTTPS domain allowlist")
        response = self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        final = urlparse(self.page.url)
        final_host = final.hostname or ""
        if final.scheme != "https" or not any(
            final_host == domain or final_host.endswith("." + domain)
            for domain in self.spec.allowed_domains
        ):
            raise ValueError("browser redirect is outside the HTTPS domain allowlist")
        if response is None or response.status >= 400:
            raise RuntimeError("asset provider returned an unsuccessful browser response")
        return self.page.locator("body").inner_text(timeout=10_000)

    def read_json(self, url: str) -> Any:
        text = self._goto_allowed(url)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("asset provider returned invalid JSON") from exc

    def close(self) -> None:
        try:
            if self.context is not None:
                self.context.close()
                self.context = None
        finally:
            if self._playwright is not None:
                self._playwright.stop()
                self._playwright = None

    def __exit__(self, *_exc: Any) -> None:
        self.close()
