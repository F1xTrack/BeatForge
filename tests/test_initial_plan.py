from beatforge import EnergyWindow, SongAnalysis, initial_plan_from_analysis, validate_plan


def test_initial_plan_from_audio_energy_is_reviewable():
    analysis = SongAnalysis("song.wav", 4.0, 8000, 1, (
        EnergyWindow(0, 2, .1, .2), EnergyWindow(2, 4, .4, .8),
    ))
    plan = initial_plan_from_analysis(analysis, bpm=120, target_nps=2.5, section_beats=2)
    assert len(plan.sections) == 4
    assert plan.sections[0].energy < plan.sections[-1].energy
    assert validate_plan(plan) == []
