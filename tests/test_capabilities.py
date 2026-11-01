import pytest

from beatforge import CapabilityProfile, effective_capabilities, validate_capabilities


def test_mod_toggles_are_independent():
    profile = effective_capabilities("pc", chroma=True, noodle=False,
                                     mapping_extensions=True, vivify=True)
    assert profile.chroma is True
    assert profile.noodle is False
    assert profile.mapping_extensions is True
    assert profile.vivify is True
    assert validate_capabilities(profile) == []


def test_vivify_quest_is_rejected_until_runtime_support_exists():
    with pytest.raises(ValueError, match="PC-only"):
        effective_capabilities("quest", vivify=True)
