from beatforge import render_preview_html


def test_preview_contains_native_objects_and_escapes_title():
    html = render_preview_html({
        "colorNotes": [{"b": 1, "x": 1, "y": 0, "c": 0, "d": 1}],
        "sliders": [{"b": 1, "x": 1, "y": 0, "c": 0, "tb": 2, "tx": 2, "ty": 0}],
        "burstSliders": [{"b": 2, "x": 2, "y": 0, "c": 1, "tb": 3, "tx": 3, "ty": 1, "sc": 4}],
        "obstacles": [], "basicBeatmapEvents": [],
    }, title="<map>")
    assert "&lt;map&gt;" in html
    assert "chainSlices" in html
    assert "burstSliders" in html
    assert "canvas" in html
