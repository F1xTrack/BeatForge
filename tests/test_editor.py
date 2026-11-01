from beatforge import render_editor_html


def test_editor_contains_typed_script_controls():
    page = render_editor_html(default_bpm=128)
    assert 'value="128"' in page
    assert "map.arc" in page
    assert "fetch('/api/compile'" in page
