from decimal import Decimal

import pytest

from beatforge import BudgetGuard


def test_budget_guard_counts_requests_repairs_and_cost():
    guard = BudgetGuard(max_requests=3, max_repairs=1, max_cost_usd=Decimal("3"))
    guard.reserve("demo/model:free")
    guard.reserve("demo/model:free", repair=True)
    guard.reserve("demo/model:free", automatic_repair=True)
    guard.record_cost(0)
    with pytest.raises(RuntimeError, match="request budget"):
        guard.reserve("demo/model:free")
    assert guard.repairs == 1
    assert guard.automatic_repairs == 1


def test_budget_guard_default_request_limit_leaves_room_for_script_recovery_and_diagnostics():
    guard = BudgetGuard()
    assert guard.max_requests == 14


def test_budget_guard_separates_automatic_and_diagnostic_repair_limits():
    guard = BudgetGuard(max_requests=5, max_repairs=1, max_automatic_repairs=1)
    guard.reserve("demo/model:free", automatic_repair=True)
    guard.reserve("demo/model:free")
    guard.reserve("demo/model:free", repair=True)
    with pytest.raises(RuntimeError, match="repair budget"):
        guard.reserve("demo/model:free", repair=True)
    with pytest.raises(RuntimeError, match="automatic repair budget"):
        guard.reserve("demo/model:free", automatic_repair=True)
    assert guard.requests == 3


def test_budget_guard_rejects_paid_and_over_budget():
    guard = BudgetGuard(max_cost_usd=Decimal("1"))
    with pytest.raises(ValueError, match=":free"):
        guard.reserve("demo/model")
    guard.reserve("demo/model:free")
    with pytest.raises(RuntimeError, match="cost budget"):
        guard.record_cost("1.01")


def test_budget_guard_allows_loopback_local_provider():
    guard = BudgetGuard(max_requests=1)
    guard.reserve("gemma", local=True)
    assert guard.requests == 1


def test_budget_guard_allows_official_free_router():
    guard = BudgetGuard(max_requests=1)
    guard.reserve("openrouter/free")
    assert guard.requests == 1
