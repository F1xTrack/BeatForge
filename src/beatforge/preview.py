"""Dependency-free diagnostic HTML preview for a compiled V3 difficulty."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_preview_html(difficulty: dict[str, Any], *, title: str = "BeatForge preview",
                       mapping_extensions: bool = False) -> str:
    payload = json.dumps(difficulty, ensure_ascii=False, separators=(",", ":"))
    safe_title = html.escape(title)
    coordinate_expression = (
        "v>=1000?(v-1000)/1000:v<=-1000?(v+1000)/1000:v"
        if mapping_extensions else "v"
    )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>{safe_title}</title>
<style>
body {{ background:#10131a;color:#e8ecf4;font:14px system-ui;margin:24px }}
canvas {{ width:100%;height:720px;background:#181d28;border:1px solid #3d4657 }}
.legend {{display:flex;gap:18px;margin:10px 0;color:#aeb9ca}}
.dot {{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
pre {{white-space:pre-wrap;max-height:260px;overflow:auto;background:#181d28;padding:12px}}
</style>
<h1>{safe_title}</h1>
<div class="legend"><span><i class="dot" style="background:#ef536f"></i>left</span>
<span><i class="dot" style="background:#4ca6ff"></i>right</span>
<span><i class="dot" style="background:#f4c95d"></i>arc</span>
<span><i class="dot" style="background:#9e7bff"></i>chain</span></div>
<canvas id="map" width="1400" height="720"></canvas>
<pre id="summary"></pre>
<script>
const map = {payload};
const canvas = document.getElementById('map'), ctx = canvas.getContext('2d');
const notes = map.colorNotes || [], bombs = map.bombNotes || [], arcs = map.sliders || [], chains = map.burstSliders || [];
const allBeats = notes.map(n=>n.b).concat(bombs.map(n=>n.b),arcs.map(a=>a.tb),chains.map(c=>c.tb));
const end = Math.max(1, ...allBeats), x = b => 40 + (b/end)*1320;
const laneCount = {6 if mapping_extensions else 4};
const coord = v => {coordinate_expression};
const xValues = notes.map(n=>coord(n.x)).concat(bombs.map(n=>coord(n.x)),arcs.flatMap(a=>[coord(a.x),coord(a.tx)]),chains.flatMap(c=>[coord(c.x),coord(c.tx)]));
const yValues = notes.map(n=>coord(n.y)).concat(bombs.map(n=>coord(n.y)),arcs.flatMap(a=>[coord(a.y),coord(a.ty)]),chains.flatMap(c=>[coord(c.y),coord(c.ty)]));
const minLane=Math.min(0,...xValues), maxLane=Math.max(laneCount-1,...xValues), minRow=Math.min(0,...yValues), maxRow=Math.max(2,...yValues);
const laneSpan=Math.max(1,maxLane-minLane+1), rowSpan=Math.max(1,maxRow-minRow+1);
const y = (lane, row, color=0) => 24 + (((lane-minLane)*rowSpan+(maxRow-row))/(laneSpan*rowSpan))*660 + (color===0?-4:4);
ctx.strokeStyle='#31394a'; ctx.fillStyle='#8d99ac'; ctx.font='12px system-ui';
for(let lane=0; lane<laneCount; lane++) for(let row=0; row<3; row++) {{ let yy=y(lane,row); ctx.beginPath();ctx.moveTo(30,yy);ctx.lineTo(1370,yy);ctx.stroke();ctx.fillText('x'+lane+' y'+row,4,yy+4); }}
for(const n of notes) {{ ctx.fillStyle=n.c===0?'#ef536f':'#4ca6ff';ctx.beginPath();ctx.arc(x(n.b),y(n.x,n.y,n.c),7,0,Math.PI*2);ctx.fill(); }}
for(const n of bombs) {{ctx.fillStyle='#f1f3f8';ctx.beginPath();ctx.moveTo(x(n.b),y(n.x,n.y)-6);ctx.lineTo(x(n.b)+6,y(n.x,n.y));ctx.lineTo(x(n.b),y(n.x,n.y)+6);ctx.lineTo(x(n.b)-6,y(n.x,n.y));ctx.closePath();ctx.fill();}}
for(const a of arcs) {{let ax=coord(a.x),tx=coord(a.tx),ay=coord(a.y),ty=coord(a.ty);ctx.strokeStyle='#f4c95d';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(x(a.b),y(ax,ay,a.c));ctx.quadraticCurveTo((x(a.b)+x(a.tb))/2,y((ax+tx)/2,Math.max(ay,ty),a.c)-10,x(a.tb),y(tx,ty,a.tc));ctx.stroke();}}
for(const c of chains) {{let cx=coord(c.x),tx=coord(c.tx),cy=coord(c.y),ty=coord(c.ty);ctx.strokeStyle='#9e7bff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x(c.b),y(cx,cy,c.c));ctx.lineTo(x(c.tb),y(tx,ty,c.c));ctx.stroke();for(let i=0;i<c.sc;i++){{let f=i/Math.max(1,c.sc-1),t=c.b+(c.tb-c.b)*f;ctx.fillStyle='#9e7bff';ctx.beginPath();ctx.arc(x(t),y(cx+(tx-cx)*f,cy+(ty-cy)*f,c.c),4,0,Math.PI*2);ctx.fill();}}}}
document.getElementById('summary').textContent = JSON.stringify({{beats:end,notes:notes.length,bombs:bombs.length,arcs:arcs.length,chains:chains.length,chainSlices:chains.reduce((n,c)=>n+(c.sc||0),0),walls:(map.obstacles||[]).length,lights:(map.basicBeatmapEvents||[]).length}}, null, 2);
</script></html>"""


def write_preview(path: str | Path, difficulty: dict[str, Any], *, title: str = "BeatForge preview",
                  mapping_extensions: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_preview_html(
        difficulty, title=title, mapping_extensions=mapping_extensions), encoding="utf-8")
    return target
