import pytest

from beatforge import ScriptRejected, run_script
from beatforge.script_runner import (normalize_vivify_setting_categories,
                                     remove_exact_duplicate_note_calls,
                                     remove_exact_duplicate_palette_calls)


def test_script_runner_compiles_typed_calls():
    result = run_script(
        """
map.note(beat=1, x=1, y=0, color=0, direction=1)
map.arc(beat=1, color=0, x=0, y=0, direction=1, tail_beat=1.5, tail_x=2, tail_y=0, tail_color=0, tail_direction=2, curve=1)
map.chain(beat=2, color=1, x=3, y=0, direction=0, tail_beat=2.5, tail_x=2, tail_y=1, slices=4, squish=0.9)
map.custom_event(beat=3, event_type="AnimateTrack", data={"track": "portal"})
""",
        bpm=120,
    )
    assert len(result["colorNotes"]) == 3
    assert {note["b"] for note in result["colorNotes"]} == {1, 1.5}
    assert any(note["d"] == 2 and note["b"] == 1.5 for note in result["colorNotes"])
    assert result["sliders"][0]["m"] == 1
    assert result["burstSliders"][0]["sc"] == 4
    assert result["customData"]["_customEvents"][0]["t"] == "AnimateTrack"


def test_vivify_normalizer_moves_only_known_settings_to_their_schema_category():
    source = (
        "map.custom_event(beat=0, event_type='SetRenderingSettings', "
        "data={'renderSettings': {'fog': 1}, "
        "'qualitySettings': {'ambientLight': [0.1,0.2,0.3,1], 'shadowDistance': 50}})"
    )
    normalized, fixes = normalize_vivify_setting_categories(source)
    result = run_script(normalized, bpm=120)
    event = result["customData"]["_customEvents"][0]
    assert event["d"] == {
        "renderSettings": {"fog": 1, "ambientLight": [0.1, 0.2, 0.3, 1]},
        "qualitySettings": {"shadowDistance": 50},
    }
    assert fixes == [{
        "kind": "moved_vivify_setting_to_schema_category",
        "line": 1, "setting": "ambientLight",
        "from": "qualitySettings", "to": "renderSettings",
    }]


def test_vivify_normalizer_leaves_conflicts_unchanged_for_validator():
    source = (
        "map.custom_event(beat=0, event_type='SetRenderingSettings', "
        "data={'renderSettings': {'ambientLight': [0,0,0,1]}, "
        "'qualitySettings': {'ambientLight': [1,1,1,1]}})"
    )
    normalized, fixes = normalize_vivify_setting_categories(source)
    assert normalized == source
    assert fixes == []


def test_vivify_normalizer_targets_the_matching_call_on_a_shared_source_line():
    data = "{'qualitySettings': {'ambientLight': [0.1,0.2,0.3,1]}}"
    source = (
        "map.custom_event(beat=0, event_type='Other', data=" + data + "); "
        "map.custom_event(beat=0, event_type='SetRenderingSettings', data=" + data + ")"
    )
    normalized, fixes = normalize_vivify_setting_categories(source)
    events = run_script(normalized, bpm=120)["customData"]["_customEvents"]
    assert events[0]["d"] == {"qualitySettings": {"ambientLight": [0.1, 0.2, 0.3, 1]}}
    assert events[1]["d"] == {"renderSettings": {"ambientLight": [0.1, 0.2, 0.3, 1]}}
    assert len(fixes) == 1


def test_script_runner_accepts_rgb_palette_only_with_complete_environment_pair():
    result = run_script(
        "map.palette(environment_left=[150,90,210], environment_right=[70,180,140], "
        "saber_left=[40,120,255], saber_right=[255,140,20])",
        bpm=120,
    )
    assert result["customData"]["_beatforgeScenePalette"] == {
        "environment_left": [150, 90, 210], "environment_right": [70, 180, 140],
        "saber_left": [40, 120, 255], "saber_right": [255, 140, 20],
    }


@pytest.mark.parametrize("source", [
    "map.palette(environment_left=[1,2,3])",
    "map.palette(environment_left=[1,2,256], environment_right=[3,4,5])",
    "map.palette(environment_left=[1,2,3.5], environment_right=[3,4,5])",
    "map.palette([1,2,3], [3,4,5])",
])
def test_script_runner_rejects_invalid_palette(source):
    with pytest.raises((ScriptRejected, ValueError)):
        run_script(source, bpm=120)


@pytest.mark.parametrize("source", [
    "import os\nmap.note(1, 1, 0, 0, 1)",
    "map.__class__",
    "map.note(1, 1, 0, 0, 1)\nfor x in range(2):\n    pass",
])
def test_script_runner_rejects_unsafe_or_ambiguous_python(source):
    with pytest.raises(ScriptRejected):
        run_script(source, bpm=120)


def test_script_runner_summarizes_repeated_collision_diagnostics():
    source = "\n".join(
        f"map.note(beat={beat}, x=1, y=0, color=0, direction=1)\n"
        f"map.note(beat={beat}, x=1, y=0, color=0, direction=1)"
        for beat in range(4)
    )
    with pytest.raises(ValueError, match=r"duplicate_cell:.*4 occurrences.*indices 1, 3, 5, 7"):
        run_script(source, bpm=120)


@pytest.mark.parametrize("curve", [0.5, 1.0, True])
def test_script_runner_rejects_non_integer_arc_curve(curve):
    source = (
        "map.arc(beat=1, color=0, x=0, y=0, direction=1, tail_beat=2, "
        f"tail_x=1, tail_y=1, tail_color=0, curve={curve!r})"
    )
    with pytest.raises(ValueError, match="invalid_arc_curve"):
        run_script(source, bpm=120)


def test_script_runner_removes_only_exact_standalone_duplicate_of_repeat_note():
    source = (
        "map.repeat(0, 2, 4, [{'kind':'note','beat':0,'x':2,'y':0,'color':1,'direction':0}])\n"
        "map.note(beat=4, x=2, y=0, color=1, direction=0)\n"
        "map.note(beat=4, x=1, y=0, color=1, direction=1)\n"
    )
    corrected, changes = remove_exact_duplicate_note_calls(source)
    assert "map.note(beat=4, x=2, y=0, color=1, direction=0)" not in corrected
    assert "map.note(beat=4, x=1, y=0, color=1, direction=1)" in corrected
    assert len(changes) == 1
    assert changes[0]["line"] == 2
    result = run_script(corrected, bpm=120)
    assert len(result["colorNotes"]) == 3


def test_script_runner_removes_only_semantically_identical_standalone_palettes():
    source = (
        "map.palette(environment_left=[0, 0, 0], environment_right=[255, 255, 255])\n"
        "map.palette(environment_right=[255, 255, 255], environment_left=[0, 0, 0])\n"
        "map.palette(environment_left=[1, 1, 1], environment_right=[255, 255, 255])\n"
    )
    corrected, changes = remove_exact_duplicate_palette_calls(source)
    assert corrected.count("map.palette(") == 2
    assert "environment_left=[1, 1, 1]" in corrected
    assert len(changes) == 1
    assert changes[0]["kind"] == "removed_exact_duplicate_palette"
    assert changes[0]["line"] == 2


def test_script_runner_keeps_multiline_duplicate_palette_calls():
    source = (
        "map.palette(environment_left=[0, 0, 0], environment_right=[255, 255, 255])\n"
        "map.palette(\n"
        "    environment_left=[0, 0, 0], environment_right=[255, 255, 255])\n"
    )
    corrected, changes = remove_exact_duplicate_palette_calls(source)
    assert corrected == source
    assert changes == []
