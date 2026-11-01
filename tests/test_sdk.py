from beatforge import Arc, Chain, CustomEvent, MapProgram, Note, compile_program, validate_program


def test_compile_emits_native_v3_arcs_and_chains():
    program = MapProgram(
        bpm=120,
        notes=[
            Note(beat=2, x=1, y=0, color=0, direction=1),
            Note(beat=1, x=0, y=0, color=0, direction=1),
        ],
        arcs=[Arc(beat=1, color=0, x=2, y=0, direction=1,
                  tail_beat=1.5, tail_x=1, tail_y=0, tail_color=0,
                  curve=1, tail_direction=2)],
        chains=[Chain(beat=3, color=1, x=3, y=0, direction=0,
                     tail_beat=3.5, tail_x=2, tail_y=1, slices=4, squish=.9)],
    )
    data = compile_program(program)
    assert data["version"] == "3.3.0"
    assert data["colorNotes"][0]["b"] == 1
    assert {(item["b"], item["x"], item["d"]) for item in data["colorNotes"]} >= {
        (1, 2, 1), (1.5, 1, 2),
    }
    assert data["sliders"][0]["tb"] == 1.5
    assert data["burstSliders"][0]["sc"] == 4


def test_v3_arc_anchors_round_trip_as_one_typed_arc():
    from beatforge.adapters import program_from_v3

    source = MapProgram(120, arcs=[Arc(
        1, 0, 0, 0, 1, 2, 2, 1, 0, tail_direction=6,
        head_note_custom_data={"track": "arc"},
        tail_note_custom_data={"track": "arc"},
    )])
    serialized = compile_program(source)

    restored = program_from_v3(serialized, bpm=120)
    reserialized = compile_program(restored)

    assert restored.notes == []
    assert len(restored.arcs) == 1
    assert restored.arcs[0].tail_direction == 6
    assert restored.arcs[0].head_note_custom_data == {"track": "arc"}
    assert restored.arcs[0].tail_note_custom_data == {"track": "arc"}
    assert reserialized == serialized


def test_duplicate_cells_are_rejected():
    program = MapProgram(120, notes=[
        Note(1, 1, 0, 0, 1), Note(1, 1, 0, 1, 0),
    ])
    assert any(error.code == "duplicate_cell" for error in validate_program(program))


def test_compile_preserves_typed_custom_events_for_mods():
    program = MapProgram(120, custom_events=[
        CustomEvent(4, "AnimateTrack", {"track": "portal", "duration": 1}),
    ])
    data = compile_program(program)
    assert data["customData"]["_customEvents"] == [{
        "b": 4, "t": "AnimateTrack", "d": {"track": "portal", "duration": 1},
    }]
