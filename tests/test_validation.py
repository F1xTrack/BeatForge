from beatforge import Arc, Chain, MapProgram, Note, validate_program


def test_arc_curve_is_restricted_to_v3_values():
    errors = validate_program(MapProgram(120, arcs=[Arc(1, 0, 0, 0, 1, 2, 1, 0, 0, curve=2)]))
    assert any(error.code == "invalid_arc_curve" for error in errors)


def test_chain_must_have_positive_duration_and_integer_slice_count():
    chain = Chain(2, 0, 0, 0, 1, 2, 1, 0, True, 1.0)
    errors = validate_program(MapProgram(120, chains=[chain]))
    assert any(error.code == "invalid_chain_duration" for error in errors)
    assert any(error.code == "invalid_chain_slices" for error in errors)


def test_arc_head_cannot_overlap_regular_note():
    program = MapProgram(120, notes=[Note(4, 1, 0, 0, 1)],
                         arcs=[Arc(4, 0, 1, 0, 1, 6, 2, 1, 0)])
    errors = validate_program(program)
    assert any(error.code == "duplicate_cell" and "arc head overlaps" in error.message
               for error in errors)


def test_arc_tail_cannot_overlap_regular_note():
    program = MapProgram(120, notes=[Note(6, 2, 1, 0, 2)],
                         arcs=[Arc(4, 0, 1, 0, 1, 6, 2, 1, 0)])
    errors = validate_program(program)
    assert any(error.code == "duplicate_cell" and "arc tail overlaps a note" in error.message
               for error in errors)


def test_arc_tail_direction_is_validated():
    errors = validate_program(MapProgram(
        120, arcs=[Arc(1, 0, 0, 0, 1, 2, 1, 0, 0, tail_direction=9)]))
    assert any(error.code == "invalid_arc_tail_direction" for error in errors)
