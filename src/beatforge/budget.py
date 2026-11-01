"""Per-job request and spend guard."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .providers import is_free_model_id


DEFAULT_MAX_REQUESTS = 14


@dataclass(slots=True)
class BudgetGuard:
    max_requests: int = DEFAULT_MAX_REQUESTS
    max_repairs: int = 2
    max_automatic_repairs: int = 2
    max_cost_usd: Decimal = Decimal("3")
    development_free_only: bool = True
    requests: int = 0
    repairs: int = 0
    automatic_repairs: int = 0
    cost_usd: Decimal = Decimal("0")

    def reserve(self, model_id: str, *, repair: bool = False,
                automatic_repair: bool = False, local: bool = False) -> None:
        if repair and automatic_repair:
            raise ValueError("a request cannot be both a diagnostic and automatic repair")
        if self.development_free_only and not local and not is_free_model_id(model_id):
            raise ValueError("development budget permits only :free models or openrouter/free")
        if self.requests >= self.max_requests:
            raise RuntimeError("generation request budget exhausted")
        if repair and self.repairs >= self.max_repairs:
            raise RuntimeError("repair budget exhausted")
        if automatic_repair and self.automatic_repairs >= self.max_automatic_repairs:
            raise RuntimeError("automatic repair budget exhausted")
        self.requests += 1
        if repair:
            self.repairs += 1
        if automatic_repair:
            self.automatic_repairs += 1

    def record_cost(self, amount_usd: str | int | float | Decimal) -> None:
        amount = Decimal(str(amount_usd))
        if amount < 0:
            raise ValueError("cost cannot be negative")
        self.cost_usd += amount
        if self.cost_usd > self.max_cost_usd:
            raise RuntimeError("cost budget exhausted")
