import pytest

from beatforge import (Arc, Bomb, Chain, MapProgram, Note, Wall,
                       render_preview_html, run_script, validate_program)
from beatforge.mapping_extensions import (
    decode_position,
    decode_wall_height,
    decode_wall_width,
    direction_bucket,
    valid_direction,
)


@pytest.mark.parametrize(("encoded", "decoded"), [
    (1000, 0.0),
    (1125, 0.125),
    (-1000, 0.0),
    (-2125, -1.125),
    (5, 5.0),
])
def test_mapping_extensions_position_decoding(encoded, decoded):
    assert decode_position(encoded) == decoded


@pytest.mark.parametrize(("direction", "expected"), [
    (0, 0),
    (8, None),
    (1000, 0),
    (1045, 1),
    (2000, 0),
    (2360, 0),
])
def test_mapping_extensions_directions(direction, expected):
    assert valid_direction(direction, mapping_extensions=True)
    assert direction_bucket(direction, mapping_extensions=True) == expected


def test_mapping_extensions_rejects_encoded_directions_in_standard_mode():
    assert not valid_direction(1000, mapping_extensions=False)
    assert direction_bucket(1000, mapping_extensions=False) is None


def test_mapping_extensions_wall_dimensions_decode_precision_values():
    assert decode_wall_width(2000) == 1.0
    assert decode_wall_width(-2000) == -1.0
    assert decode_wall_height(3000) == 2.0
    assert decode_wall_height(-1000) == 1.0


def test_mapping_extensions_validation_preserves_encoded_playable_objects():
    program = MapProgram(
        bpm=120,
        notes=[Note(1, 1000, -2000, 0, 1180)],
        bombs=[Bomb(2, 2000, 0)],
        arcs=[Arc(3, 0, 0, 0, 1000, 4, 2000, 0, 0)],
        chains=[Chain(5, 1, 0, 0, 2000, 6, 5, 2, 3, 0.9)],
        walls=[Wall(1, 4000, 1000, 2, 2000, 3000)],
    )

    assert validate_program(program, mapping_extensions=True) == []
    errors = validate_program(program, mapping_extensions=False)
    assert any(error.code == "note_out_of_grid" for error in errors)
    assert any(error.code == "invalid_wall_geometry" for error in errors)


def test_safe_script_runner_serializes_mapping_extensions_fields_unchanged():
    result = run_script(
        "map.note(beat=1, x=1000, y=-2000, color=0, direction=1180)\n"
        "map.bomb(beat=2, x=2000, y=0)\n"
        "map.arc(beat=3, color=0, x=0, y=0, direction=1000, tail_beat=4, "
        "tail_x=2000, tail_y=0, tail_color=0, curve=0)\n"
        "map.chain(beat=5, color=1, x=0, y=0, direction=2000, tail_beat=6, "
        "tail_x=5, tail_y=2, slices=3, squish=0.9)\n"
        "map.wall(beat=1, x=4000, y=1000, duration=2, width=2000, height=3000)",
        bpm=120,
        mapping_extensions=True,
    )

    assert result["colorNotes"][0]["x"] == 1000
    assert result["colorNotes"][0]["d"] == 1180
    assert len(result["bombNotes"]) == 1
    assert len(result["sliders"]) == len(result["burstSliders"]) == 1
    assert result["obstacles"][0]["w"] == 2000


def test_preview_decodes_encoded_coordinates_and_displays_bombs():
    html = render_preview_html({
        "colorNotes": [{"b": 1, "x": 1000, "y": 0, "c": 0, "d": 0}],
        "bombNotes": [{"b": 2, "x": 2000, "y": 0}],
        "sliders": [], "burstSliders": [], "obstacles": [],
        "basicBeatmapEvents": [],
    }, mapping_extensions=True)

    assert "v>=1000?(v-1000)/1000" in html
    assert "bombs:bombs.length" in html
