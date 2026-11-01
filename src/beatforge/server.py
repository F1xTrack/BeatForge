"""Local-only HTTP preview server."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Any

from .preview import render_preview_html
from .script_runner import ScriptRejected, run_script
from .editor import render_editor_html


def create_preview_server(difficulty: dict[str, Any], *, title: str = "BeatForge preview",
                          host: str = "127.0.0.1", port: int = 0,
                          mapping_extensions: bool = False) -> ThreadingHTTPServer:
    page = render_preview_html(
        difficulty, title=title, mapping_extensions=mapping_extensions).encode("utf-8")
    editor_page = render_editor_html().encode("utf-8")
    health = json.dumps({"status": "ok", "service": "beatforge-preview"}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib handler API
            if self.path == "/health":
                body, content_type = health, "application/json; charset=utf-8"
            elif self.path in {"/", "/index.html"}:
                body, content_type = page, "text/html; charset=utf-8"
            elif self.path == "/editor":
                body, content_type = editor_page, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802 - stdlib handler API
            if self.path != "/api/compile":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > 512 * 1024:
                    raise ValueError("request body exceeds 512 KiB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("script"), str):
                    raise ValueError("JSON body requires a script string")
                result = run_script(payload["script"], bpm=payload.get("bpm"),
                                    mapping_extensions=mapping_extensions)
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                status = 200
            except (ValueError, TypeError, json.JSONDecodeError, ScriptRejected) as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve_preview(difficulty: dict[str, Any], *, title: str = "BeatForge preview",
                  host: str = "127.0.0.1", port: int = 8765,
                  mapping_extensions: bool = False) -> None:
    server = create_preview_server(difficulty, title=title, host=host, port=port,
                                   mapping_extensions=mapping_extensions)
    try:
        server.serve_forever()
    finally:
        server.server_close()
