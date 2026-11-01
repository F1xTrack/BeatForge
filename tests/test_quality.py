from dataclasses import replace

from beatforge import (Arc, Chain, LightEvent, MapProgram, Note, Section, TrackPlan,
                       preflight)


def _audio_grid(levels, attack_levels=None):
    rows = []
    for index, level in enumerate(levels):
        start = index * 8
        values = str(level) * 16
        attack = str(attack_levels[index]) * 16 if attack_levels is not None else values
        rows.append(f"beats {start}-{start + 8} low={values} mid={values} "
                    f"high={values} attack={attack}")
    return "Measured local grid\n" + "\n".join(rows)


def _repeated_phrase(levels, *, vary_by_audio=False):
    notes = []
    for window in range(len(levels)):
        direction = (window % 4) if vary_by_audio else 1
        x = direction if vary_by_audio else 1
        for offset in (0, 2, 4, 6):
            beat = window * 8 + offset
            notes.append(Note(float(beat), x, 0, 0, direction))
    return MapProgram(120, notes=notes)


def test_preflight_counts_chain_slices_and_sections():
    plan = TrackPlan(120, 10, 2.5, sections=[Section(0, 4, "intro"), Section(4, 8, "drop")])
    program = MapProgram(120,
                         notes=[Note(1, 1, 0, 0, 1)],
                         chains=[Chain(4, 1, 3, 0, 0, 5, 2, 1, 4, .9)])
    report = preflight(program, plan)
    assert report.passed is True
    assert report.chain_slices == 4
    assert report.note_count == 5
    assert report.occupied_sections == 2


def test_preflight_blocks_unfilled_active_planned_sections():
    plan = TrackPlan(120, 10, 1.0, sections=[
        Section(0, 4, "intro", density=1.0),
        Section(4, 8, "drop", density=2.0),
    ])
    program = MapProgram(120, notes=[Note(1, 1, 0, 0, 1)])

    report = preflight(program, plan)

    assert report.passed is False
    assert report.occupied_sections == 1
    assert any("active planned section 'drop' has no authored playable object"
               in error for error in report.errors)
    assert any("not all active planned sections contain playable objects (1/2 occupied)"
               in error for error in report.errors)


def test_preflight_allows_empty_zero_density_and_locked_sections():
    plan = TrackPlan(120, 3, 0.5, sections=[
        Section(0, 2, "active", density=0.5),
        Section(2, 4, "rest", density=0.0),
        Section(4, 6, "locked-rest", density=0.5, locked=True),
    ])
    program = MapProgram(120, notes=[Note(1, 1, 0, 0, 1)])

    report = preflight(program, plan)

    assert report.passed is True
    assert report.occupied_sections == 1
    assert not any("planned section" in error for error in report.errors)


def test_preflight_warns_when_environment_palette_colors_are_too_similar():
    plan = TrackPlan(120, 10, 2.5)
    program = MapProgram(120, scene_palette={
        "environment_left": [22, 12, 44],
        "environment_right": [80, 24, 110],
    })

    report = preflight(program, plan)

    assert report.palette_contrast["available"] is True
    assert report.palette_contrast["warning"] is True
    assert report.palette_contrast["normalized_rgb_distance"] < 0.30
    assert any("scene_palette_low_contrast" in warning for warning in report.warnings)


def test_preflight_accepts_visibly_separated_environment_palette_colors():
    plan = TrackPlan(120, 10, 2.5)
    program = MapProgram(120, scene_palette={
        "environment_left": [35, 125, 245],
        "environment_right": [245, 115, 25],
    })

    report = preflight(program, plan)

    assert report.palette_contrast["warning"] is False
    assert not any("scene_palette_low_contrast" in warning for warning in report.warnings)


def test_preflight_flags_flat_map_density_when_spectral_activity_changes():
    levels = [9, 0, 9, 0, 9, 0]
    plan = TrackPlan(120, 24, 2.5)
    notes = [Note(window * 8 + offset, int(offset * 2) % 4, 0,
                  (window + int(offset * 2)) % 2, 1)
             for window in range(6) for offset in [i * 0.5 for i in range(16)]]

    report = preflight(MapProgram(120, notes=notes), plan,
                       audio_features=_audio_grid(levels))

    assert report.rhythm_alignment["available"] is True
    assert report.rhythm_alignment["warning"] is True
    assert report.rhythm_alignment["map_density_cv"] == 0
    assert report.rhythm_alignment["audio_activity_cv"] > 0.75
    assert report.rhythm_alignment["density_activity_correlation"] is None
    assert any("rhythm_density_unresponsive" in warning for warning in report.warnings)


def test_preflight_keeps_exact_half_beat_spectral_change_candidates():
    plan = TrackPlan(120, 24, 2.5)
    report = preflight(
        MapProgram(120, notes=[Note(beat, beat % 4, 0, 0, 1)
                               for beat in range(0, 48, 2)]),
        plan, audio_features=_audio_grid([9, 0, 9, 0, 9, 0], [9, 0, 9, 0, 9, 0]),
    )

    profiles = report.rhythm_alignment["window_profiles"]
    assert profiles[0]["strong_spectral_change_beats"] == [
        round(index * 0.5, 3) for index in range(16)
    ]
    assert profiles[1]["strong_spectral_change_beats"] == []


def test_preflight_accepts_density_that_tracks_large_audio_activity_changes():
    levels = [9, 0, 9, 0, 9, 0]
    plan = TrackPlan(120, 24, 2.5)
    notes = []
    for window, level in enumerate(levels):
        offsets = [i * 0.5 for i in range(16)] if level >= 8 else [0, 2, 4, 6]
        notes.extend(Note(window * 8 + offset, int(offset) % 4, 0,
                          (window + int(offset)) % 2, 1) for offset in offsets)

    report = preflight(MapProgram(120, notes=notes), plan,
                       audio_features=_audio_grid(levels))

    assert report.rhythm_alignment["warning"] is False
    assert report.rhythm_alignment["density_activity_correlation"] > 0.9


def test_preflight_distinguishes_sustained_high_energy_from_quiet_low_attack_windows():
    levels = [9, 9, 9, 0, 0, 0]
    plan = TrackPlan(120, 24, 2.5)
    notes = [Note(window * 8 + offset, int(offset) % 4, 0,
                  (window + int(offset)) % 2, 1)
             for window in range(6) for offset in (0, 2, 4, 6)]

    report = preflight(MapProgram(120, notes=notes), plan,
                       audio_features=_audio_grid(levels, [0] * 6))

    rhythm = report.rhythm_alignment
    assert rhythm["warning"] is True
    assert rhythm["window_profiles"][0]["audio_strong_attack_cells"] == 0
    assert rhythm["window_profiles"][0]["audio_character"] == "sustained"
    assert rhythm["window_profiles"][3]["audio_character"] == "quiet"
    assert rhythm["window_profiles"][0]["audio_activity"] > rhythm["window_profiles"][3]["audio_activity"]
    assert rhythm["window_profiles"][0]["repair_target"] is True


def test_preflight_accepts_density_that_tracks_sustained_energy_without_attacks():
    levels = [9, 9, 9, 0, 0, 0]
    plan = TrackPlan(120, 24, 2.5)
    notes = []
    for window, level in enumerate(levels):
        count = 8 if level else 6
        notes.extend(Note(window * 8 + offset, offset % 4, 0,
                          (window + offset) % 2, 1)
                     for offset in range(count))

    report = preflight(MapProgram(120, notes=notes), plan,
                       audio_features=_audio_grid(levels, [0] * 6))

    rhythm = report.rhythm_alignment
    assert rhythm["audio_attack_cv"] == 0
    assert rhythm["audio_band_energy_cv"] >= 0.75
    assert rhythm["density_activity_correlation"] > 0.9
    assert rhythm["warning"] is False


def test_preflight_exposes_three_or_more_rapid_same_hand_hits_as_chain_opportunity():
    plan = TrackPlan(120, 16, 2.0)
    notes = [Note(39.5 + index * 0.25, index % 4, 1, 1, index % 8)
             for index in range(8)]

    report = preflight(MapProgram(120, notes=notes), plan)

    assert report.chain_opportunities["warning"] is True
    assert report.chain_opportunities["review_recommended"] is True
    assert report.chain_opportunities["actionable_repair"] is True
    assert report.chain_opportunities["eligible_run_count"] == 1
    assert report.chain_opportunities["eligible_runs"] == [{
        "color": 1,
        "start_beat": 39.5,
        "end_beat": 41.25,
        "head_count": 8,
        "max_gap_beats": 0.25,
    }]
    assert any("native_chain_candidate_review" in warning for warning in report.warnings)
    assert "no chain quota" in next(
        warning for warning in report.warnings
        if "native_chain_candidate_review" in warning)


def test_pattern_repair_targets_include_bounded_authored_event_sequences():
    plan = TrackPlan(120, 16, 1.0)
    notes = [Note(window * 8 + offset, 0, 0, 0, 1)
             for window in range(4) for offset in (0, 2, 4, 6)]

    report = preflight(MapProgram(120, notes=notes), plan,
                       audio_features=_audio_grid([1, 1, 8, 2]))
    targets = report.pattern_alignment["repair_targets"]

    assert targets
    signature_target = targets[0]
    assert len(signature_target["window_signatures"]) == 2
    left, right = signature_target["window_signatures"]
    assert left["window"] == signature_target["between_windows"][0]
    assert right["window"] == signature_target["between_windows"][1]
    assert left["total_events"] == 4
    assert left["events"][0] == {
        "beat_offset": 0.0, "hand": "left", "x": 0.0, "y": 0.0,
        "direction_bucket": 1, "type": "note",
    }
    assert right["events"][0]["beat_offset"] == 0.0
    assert right["truncated"] is False


def test_chain_usage_warns_when_slices_dominate_playable_hit_equivalents():
    notes = [Note(float(index), index % 4, 0, index % 2, index % 8)
             for index in range(200)]
    chains = [Chain(200 + index * 2, index % 2, index % 4, 0, index % 8,
                    201 + index * 2, (index + 1) % 4, 0, 4, 0.9)
              for index in range(50)]

    report = preflight(MapProgram(90, notes=notes, chains=chains),
                       TrackPlan(90, 200, 2.0))

    assert report.chain_usage["chain_slices"] == 200
    assert report.chain_usage["total_hit_equivalents"] == 400
    assert report.chain_usage["chain_slice_share"] == 0.5
    assert report.chain_usage["warning_threshold"] == 0.35
    assert report.chain_usage["warning"] is True
    assert any("chain_slice_overuse" in warning for warning in report.warnings)


def test_chain_usage_does_not_warn_at_same_track_reference_shares():
    notes = [Note(float(index), index % 4, 0, index % 2, index % 8)
             for index in range(406)]
    chains = [Chain(410 + index * 2, index % 2, index % 4, 0, index % 8,
                    410.0833 + index * 2, (index + 1) % 4, 0, 4, 0.9)
              for index in range(16)]

    report = preflight(MapProgram(90, notes=notes, chains=chains),
                       TrackPlan(90, 320, 2.0))

    assert report.chain_usage["chain_slices"] == 64
    assert report.chain_usage["chain_slice_share"] == round(64 / 470, 6)
    assert report.chain_usage["warning"] is False
    assert not any("chain_slice_overuse" in warning for warning in report.warnings)


def test_preflight_does_not_call_slow_or_alternating_hands_chain_candidates():
    plan = TrackPlan(120, 16, 2.0)
    notes = [Note(float(index), index % 4, 1, index % 2, 1)
             for index in range(8)]

    report = preflight(MapProgram(120, notes=notes), plan)

    assert report.chain_opportunities["eligible_runs"] == []
    assert report.chain_opportunities["warning"] is False
    assert not any("native_chain_candidate_review" in warning for warning in report.warnings)


def test_preflight_does_not_recommend_frequent_two_hit_pairs_as_chains():
    plan = TrackPlan(120, 16, 2.0)
    notes = [Note(0.0, 0, 1, 1, 1), Note(0.25, 1, 1, 1, 2),
             Note(4.0, 2, 1, 1, 3), Note(4.25, 3, 1, 1, 4)]

    report = preflight(MapProgram(120, notes=notes), plan)

    assert report.chain_opportunities["eligible_run_count"] == 0
    assert report.chain_opportunities["warning"] is False


def test_preflight_keeps_unconverted_note_run_informational_when_other_chains_exist():
    plan = TrackPlan(120, 16, 2.0)
    notes = [Note(index * 0.25, index % 4, 1, 1, index % 8)
             for index in range(4)]
    chain = Chain(8, 0, 1, 0, 1, 9, 2, 1, 4, 0.9)

    report = preflight(MapProgram(120, notes=notes, chains=[chain]), plan)

    assert report.chain_opportunities["eligible_run_count"] == 1
    assert report.chain_opportunities["chains_present"] == 1
    assert report.chain_opportunities["unrepresented_run_count"] == 1
    assert report.chain_opportunities["warning"] is True
    assert report.chain_opportunities["review_recommended"] is True
    assert report.chain_opportunities["actionable_repair"] is True


def test_preflight_warns_when_lights_skip_most_playable_sections():
    plan = TrackPlan(120, 16, 1.0, sections=[
        Section(0, 4, "one"), Section(4, 8, "two"),
        Section(8, 12, "three"), Section(12, 16, "four"),
    ])
    program = MapProgram(
        120,
        notes=[Note(1, 0, 0, 0, 1), Note(5, 0, 0, 0, 1),
               Note(9, 0, 0, 0, 1), Note(13, 0, 0, 0, 1)],
        lights=[LightEvent(1, 0, 1)],
    )

    report = preflight(program, plan)

    assert report.lighting_coverage["active_sections"] == ["one", "two", "three", "four"]
    assert report.lighting_coverage["covered_sections"] == ["one"]
    assert report.lighting_coverage["coverage_ratio"] == 0.25
    assert report.lighting_coverage["warning"] is True
    assert any("lighting_section_coverage_low" in warning for warning in report.warnings)


def test_preflight_accepts_lights_covering_half_of_active_sections_without_event_quota():
    plan = TrackPlan(120, 16, 1.0, sections=[
        Section(0, 4, "one"), Section(4, 8, "two"),
        Section(8, 12, "three"), Section(12, 16, "four"),
    ])
    program = MapProgram(
        120,
        notes=[Note(1, 0, 0, 0, 1), Note(5, 0, 0, 0, 1),
               Note(9, 0, 0, 0, 1), Note(13, 0, 0, 0, 1)],
        lights=[LightEvent(1, 0, 1), LightEvent(9, 0, 1)],
    )

    report = preflight(program, plan)

    assert report.lighting_coverage["coverage_ratio"] == 0.5
    assert report.lighting_coverage["warning"] is False
    assert not any("lighting_section_coverage_low" in warning for warning in report.warnings)


def test_preflight_fails_duplicate_notes():
    plan = TrackPlan(120, 10, 2.5)
    program = MapProgram(120, notes=[Note(1, 1, 0, 0, 1), Note(1, 1, 0, 1, 0)])
    assert preflight(program, plan).passed is False


def test_preflight_counts_both_arc_endpoint_hits_and_arc_only_section():
    plan = TrackPlan(120, 10, 1.0, sections=[Section(0, 12, "arc-section")])
    program = MapProgram(120, arcs=[Arc(8, 0, 0, 0, 1, 12, 1, 0, 0)])
    report = preflight(program, plan)
    assert report.note_count == 2
    assert report.occupied_sections == 1


def test_preflight_warns_on_unvaried_one_beat_note_run():
    plan = TrackPlan(120, 16, 2.0)
    program = MapProgram(120, notes=[Note(float(beat), 1, 0, 0, 1) for beat in range(10)])
    report = preflight(program, plan)
    assert any("same grid cell every beat for 10 notes" in warning for warning in report.warnings)


def test_preflight_uses_full_track_duration_for_nps():
    plan = TrackPlan(120, 10, 1.0)
    program = MapProgram(120, notes=[Note(1, 1, 0, 0, 1)])
    report = preflight(program, plan)
    assert report.actual_nps == 0.1
    assert any("actual NPS 0.100" in warning for warning in report.warnings)


def test_preflight_flags_nps_outside_tighter_fifteen_percent_band():
    plan = TrackPlan(90, 100, 3.5)
    notes = [Note(index * 0.35, index % 4, (index // 4) % 3,
                  index % 2, index % 8) for index in range(425)]

    report = preflight(MapProgram(90, notes=notes), plan)

    assert report.actual_nps == 4.25
    assert any("outside the ±15% target band" in warning for warning in report.warnings)


def test_preflight_warns_only_for_extreme_per_hand_direction_monotony():
    plan = TrackPlan(120, 30, 2.5)
    notes = [Note(float(i), i % 4, i % 3, 0, 1) for i in range(30)]
    notes += [Note(float(i) + 0.5, i % 4, i % 3, 1, i % 4) for i in range(30)]

    report = preflight(MapProgram(120, notes=notes), plan)

    assert report.direction_profiles["0"] == {
        "playable_heads": 30,
        "direction_counts": {"1": 30},
        "dominant_direction": 1,
        "dominant_fraction": 1.0,
    }
    assert any("severe cut-direction monotony in color 0 hand" in warning
               and "30/30 playable heads (100%)" in warning for warning in report.warnings)
    assert not any("severe cut-direction monotony in color 1 hand" in warning
                   for warning in report.warnings)


def test_preflight_flags_repeated_pattern_across_changing_audio_sections():
    levels = [1, 9, 1, 9, 1, 9, 1, 9]
    plan = TrackPlan(120, 32, 2.0, sections=[
        Section(0, 16, "quiet", energy=0.2), Section(16, 32, "loud", energy=0.9),
    ])
    report = preflight(_repeated_phrase(levels), plan, audio_features=_audio_grid(levels))

    assert report.pattern_alignment["available"] is True
    assert report.pattern_alignment["warning"] is True
    assert report.pattern_alignment["warning_threshold"] == 0.20
    assert report.pattern_alignment["chance_corrected_structure_support"] < 0.05
    assert any("map pattern changes show little relationship to measured audio changes" in warning
               for warning in report.warnings)


def test_pattern_diagnostic_exposes_actionable_four_second_profiles_and_transitions():
    levels = [1, 1, 9, 9, 1, 1, 9, 9]
    plan = TrackPlan(120, 32, 2.0)
    report = preflight(_repeated_phrase(levels), plan, audio_features=_audio_grid(levels))

    profiles = report.pattern_alignment["window_profiles"]
    assert len(profiles) == 8
    assert profiles[0]["beats"] == [0.0, 8.0]
    assert profiles[0]["map_heads"] == 4
    assert profiles[0]["map_heads_by_type"] == {
        "notes": 4, "arc_heads": 0, "chain_heads": 0, "bombs": 0,
    }
    assert profiles[0]["hands_left_right"] == [1.0, 0.0]
    assert profiles[2]["audio_mean_0_9"] == [9.0, 9.0, 9.0]
    targets = report.pattern_alignment["repair_targets"]
    assert targets
    assert len(targets) <= 4
    assert all(target["audio_change"] > target["map_change"] for target in targets)
    assert all(target["unmatched_change"] > 0 for target in targets)


def test_preflight_does_not_flag_map_patterns_that_follow_audio_changes():
    levels = [1, 3, 7, 9, 1, 3, 7, 9]
    plan = TrackPlan(120, 32, 2.0, sections=[
        Section(0, 16, "build", energy=0.3), Section(16, 32, "release", energy=0.9),
    ])
    report = preflight(_repeated_phrase(levels, vary_by_audio=True), plan,
                       audio_features=_audio_grid(levels))

    assert report.pattern_alignment["available"] is True
    assert report.pattern_alignment["warning"] is False


def test_preflight_reports_pattern_alignment_unavailable_without_audio_features():
    plan = TrackPlan(120, 32, 2.0)
    report = preflight(_repeated_phrase([1, 9, 1, 9]), plan)

    assert report.pattern_alignment == {
        "available": False, "reason": "audio feature grid unavailable",
    }
    assert not any("pattern changes show little relationship" in warning
                   for warning in report.warnings)


def test_chain_slice_count_does_not_dominate_phrase_pattern_alignment():
    levels = [1, 3, 7, 9, 1, 3, 7, 9]
    plan = TrackPlan(120, 32, 2.0)
    base = _repeated_phrase(levels, vary_by_audio=True)
    short_chains = [Chain(float(window * 8 + 1), 1, 3, 0, 1,
                          float(window * 8 + 3), 2, 1, 2, 0.9)
                    for window in range(8)]
    dense_chains = [replace(chain, slices=16) for chain in short_chains]

    short = preflight(replace(base, chains=short_chains), plan,
                      audio_features=_audio_grid(levels))
    dense = preflight(replace(base, chains=dense_chains), plan,
                      audio_features=_audio_grid(levels))

    assert short.chain_slices == 16
    assert dense.chain_slices == 128
    assert short.pattern_alignment == dense.pattern_alignment
