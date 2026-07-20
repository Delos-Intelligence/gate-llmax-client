"""Build a ``call_prefer([...])`` fallback list that covers every plan an app serves.

The problem: a cosmos-style app has users on different *plans* (hosting-provider cost tiers).
A chat call sets ``plan=<the user's plan>``, which filters routing to that plan's providers —
so a given model may resolve on some plans and 404 / NO_DEPLOYMENT on others. To make one call
work for every user regardless of plan, pass an ordered ``call_prefer([...])`` list: the gateway
tries the models in order and the first that has a deployment on the caller's plan wins.

``build_prefer_list`` turns the model×plan matrix into such a list: preferred models first
(quality), then the minimum set of broadly-available models needed so every target plan is
covered by *some* entry in the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gate_llmax.models.config import ModelPlanRow, ModelPurpose


@dataclass(frozen=True)
class PreferStep:
    """One model appended to the fallback list and the target plans it is the first to cover."""

    model: str
    covers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreferResult:
    """The built fallback list plus how it maps onto the target plans."""

    models: list[str]
    """Ordered model names — pass straight to ``.call_prefer(models)``."""
    coverage: dict[str, str | None]
    """plan_id → the first model in ``models`` that is reachable on it (None if none is)."""
    uncovered_plans: list[str]
    """Target plans no model of this purpose can serve — calls with these plans will fail."""
    steps: list[PreferStep]
    """Why each model is in the list (the plans it first covers). Empty ``covers`` = a preferred seed."""

    def snippet(self, operation: str = "my-operation") -> str:
        """A ready-to-paste ``call_prefer`` call for this list."""
        listing = ", ".join(f'"{m}"' for m in self.models)
        return f'resp = await client.request(prompt=prompt, operation="{operation}").call_prefer([{listing}])'


def build_prefer_list(
    matrix: list[ModelPlanRow],
    *,
    purpose: ModelPurpose | str = ModelPurpose.CHAT,
    plans: list[str] | None = None,
    prefer: list[str] | None = None,
) -> PreferResult:
    """Build a plan-covering ``call_prefer`` list from the model×plan availability ``matrix``.

    Args:
        matrix: rows from ``client.model_plan_matrix()`` (model → the plans it is reachable on).
        purpose: only consider models of this purpose (``chat`` by default).
        plans: the plans to cover, in fallback-tail priority order (e.g. premium→cheap). ``None``
            covers every plan any model of this purpose serves, ordered alphabetically — pass the
            ids from ``client.list_plans()`` (already sorted by ``sort_order``) for the real order.
        prefer: model names to place first, in order — the quality ranking. Names absent from the
            matrix for this purpose are skipped. Coverage is still guaranteed by appending broad
            models after the preferred ones.

    Returns:
        A ``PreferResult``. ``models`` is the ordered list; whatever plan a caller is on, the chain
        reaches a working model unless that plan is in ``uncovered_plans``.
    """
    purpose_value = purpose.value if isinstance(purpose, ModelPurpose) else str(purpose)
    reachable: dict[str, set[str]] = {
        row.model_name: set(row.available_plan_ids)
        for row in matrix
        if (row.purpose.value if isinstance(row.purpose, ModelPurpose) else str(row.purpose)) == purpose_value
    }

    # Caller order (de-duped) when plans are given; else every plan any model serves, alphabetical.
    target_plans = list(dict.fromkeys(plans)) if plans is not None else sorted({p for covered in reachable.values() for p in covered})

    # Broadest coverage first so the fallback tail is a robust catch-all; alphabetical tie-break.
    def breadth(model: str) -> tuple[int, str]:
        return (-len(reachable[model]), model)

    models: list[str] = []
    steps: list[PreferStep] = []

    def covered_plans() -> set[str]:
        return {p for m in models for p in reachable.get(m, set())}

    for model in prefer or []:
        if model in reachable and model not in models:
            models.append(model)
            steps.append(PreferStep(model=model, covers=[]))

    uncovered: list[str] = []
    for plan in target_plans:
        if plan in covered_plans():
            continue
        candidates = sorted((m for m, ps in reachable.items() if plan in ps and m not in models), key=breadth)
        if not candidates:
            uncovered.append(plan)
            continue
        chosen = candidates[0]
        newly = [p for p in target_plans if p in reachable[chosen] and p not in covered_plans()]
        models.append(chosen)
        steps.append(PreferStep(model=chosen, covers=newly))

    coverage: dict[str, str | None] = {}
    for plan in target_plans:
        coverage[plan] = next((m for m in models if plan in reachable.get(m, set())), None)

    return PreferResult(models=models, coverage=coverage, uncovered_plans=uncovered, steps=steps)
