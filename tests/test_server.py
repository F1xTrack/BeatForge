import json
from threading import Thread
from urllib.request import Request, urlopen

from beatforge import create_preview_server


def test_local_preview_server_serves_health_and_html():
    server = create_preview_server({"colorNotes": [], "sliders": [], "burstSliders": []}, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/health", timeout=2) as response:
            assert json.loads(response.read())["status"] == "ok"
        with urlopen(base + "/", timeout=2) as response:
            assert b"<canvas" in response.read()
        with urlopen(base + "/editor", timeout=2) as response:
            assert b"/api/compile" in response.read()
        payload = b'{"bpm":120,"script":"map.note(1, 1, 0, 0, 1)"}'
        request = Request(base + "/api/compile", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=2) as response:
            assert json.loads(response.read())["colorNotes"][0]["b"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
