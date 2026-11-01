"""Small browser editor shell for the local compile endpoint."""
from __future__ import annotations

import html


def render_editor_html(*, default_bpm: float = 120.0) -> str:
    bpm = html.escape(str(default_bpm))
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>BeatForge editor</title>
<style>body{{background:#10131a;color:#e8ecf4;font:14px system-ui;margin:24px;max-width:1000px}}textarea{{width:100%;height:260px;background:#181d28;color:#e8ecf4;border:1px solid #3d4657;padding:12px;font:13px monospace}}input,button{{margin:8px 8px 8px 0;padding:8px;background:#252d3d;color:#e8ecf4;border:1px solid #57657a}}pre{{white-space:pre-wrap;background:#181d28;padding:12px;min-height:80px}}</style>
<h1>BeatForge editor</h1>
<label>BPM <input id="bpm" type="number" min="1" step="0.01" value="{bpm}"></label>
<textarea id="script">map.arc(beat=1, color=0, x=1, y=0, direction=1, tail_beat=1.5, tail_x=2, tail_y=0, tail_color=0, tail_direction=2, curve=1)</textarea>
<br><button id="compile">Compile V3</button><pre id="output">Ready.</pre>
<script>
document.getElementById('compile').onclick = async () => {{
  const output = document.getElementById('output');
  output.textContent = 'Compiling...';
  try {{
    const response = await fetch('/api/compile', {{method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{bpm:Number(document.getElementById('bpm').value), script:document.getElementById('script').value}})}});
    const value = await response.json();
    output.textContent = JSON.stringify(value, null, 2);
  }} catch (error) {{ output.textContent = String(error); }}
}};
</script></html>"""
