"""Exception narration.

The engine computes; the model writes the sentence. The constraint that makes
this safe rather than decorative: **every number in the output must appear in the
input payload.** A model cannot introduce a figure, so it cannot invent a
plausible-sounding amount, gap, or date.

The diagnosis itself is arithmetic and happens before any model is called.
"""

import pytest

from allocation_agent.decide.narrate import (
    NarrationError,
    Narrator,
    StubBackend,
    diagnose_residual,
    validate_numbers,
)

# --------------------------------------------------------------------------- #
# residual diagnosis is arithmetic, not language
# --------------------------------------------------------------------------- #

def test_gap_matching_a_known_fee_rate_is_diagnosed_as_a_charge():
    causes = diagnose_residual(residual_minor=200, amount_minor=20_000,
                               n_lines=1, usual_fee_bps=100)
    assert causes[0][0] == "BANK_CHARGE"


def test_tiny_gap_scaled_by_line_count_is_diagnosed_as_rounding():
    causes = diagnose_residual(residual_minor=3, amount_minor=500_000,
                               n_lines=4, usual_fee_bps=0)
    assert causes[0][0] == "ROUNDING"


def test_zero_residual_is_not_an_exception():
    assert diagnose_residual(residual_minor=0, amount_minor=1000,
                             n_lines=1, usual_fee_bps=0) == []


def test_every_cause_is_scored_so_the_ranking_is_inspectable():
    causes = diagnose_residual(residual_minor=999, amount_minor=100_000,
                               n_lines=2, usual_fee_bps=50)
    assert len(causes) >= 3
    assert causes == sorted(causes, key=lambda c: -c[1])


def test_an_unexplained_gap_still_returns_a_ranking_with_low_fit():
    causes = diagnose_residual(residual_minor=123_456, amount_minor=1000,
                               n_lines=1, usual_fee_bps=0)
    assert causes[0][1] < 0.5


# --------------------------------------------------------------------------- #
# the model may not introduce a number
# --------------------------------------------------------------------------- #

def test_a_sentence_using_only_supplied_figures_passes():
    validate_numbers("A gap of 2.00 against an amount of 200.00.",
                     allowed={"2.00", "200.00", "2", "200"})


def test_an_invented_figure_is_rejected():
    with pytest.raises(NarrationError, match="not in the payload"):
        validate_numbers("A bank charge of 7.50 was deducted.", allowed={"2.00"})


def test_the_rejection_names_the_offending_figure():
    with pytest.raises(NarrationError, match=r"7\.50"):
        validate_numbers("charge of 7.50", allowed={"2.00"})


def test_text_with_no_figures_is_fine():
    validate_numbers("The counterparty could not be identified.", allowed=set())


def test_percentages_and_ids_are_checked_too():
    with pytest.raises(NarrationError):
        validate_numbers("matched invoice 4471", allowed={"2.00"})


# --------------------------------------------------------------------------- #
# the narrator degrades rather than failing
# --------------------------------------------------------------------------- #

def test_stub_backend_produces_a_usable_sentence_without_any_api():
    n = Narrator(backend=StubBackend())
    out = n.narrate([{"record_id": "b1", "residual_minor": 200, "amount_minor": 20_000,
                      "causes": [("BANK_CHARGE", 0.98)]}])
    assert out[0]["cause"] == "BANK_CHARGE"
    assert out[0]["sentence"]


def test_a_failing_backend_falls_back_to_a_template_rather_than_raising():
    class Broken:
        def complete(self, prompt):
            raise RuntimeError("service unavailable")

    n = Narrator(backend=Broken())
    out = n.narrate([{"record_id": "b1", "residual_minor": 200, "amount_minor": 20_000,
                      "causes": [("BANK_CHARGE", 0.98)]}])
    assert out[0]["source"] == "template"
    assert out[0]["cause"] == "BANK_CHARGE"


def test_a_backend_that_invents_a_number_is_rejected_and_falls_back():
    class Liar:
        def complete(self, prompt):
            return '[{"record_id":"b1","cause":"BANK_CHARGE","sentence":"Fee of 99.99 applied."}]'

    n = Narrator(backend=Liar())
    out = n.narrate([{"record_id": "b1", "residual_minor": 200, "amount_minor": 20_000,
                      "causes": [("BANK_CHARGE", 0.98)]}])
    assert out[0]["source"] == "template", "invented figure must not reach the output"


def test_narration_is_batched_into_one_call():
    calls = []

    class Counting:
        def complete(self, prompt):
            calls.append(prompt)
            return "[]"

    n = Narrator(backend=Counting(), batch_size=20)
    n.narrate([{"record_id": f"b{i}", "residual_minor": 1, "amount_minor": 100,
                "causes": [("ROUNDING", 0.9)]} for i in range(20)])
    assert len(calls) == 1


def test_repeated_situations_are_not_re_narrated():
    calls = []

    class Counting:
        def complete(self, prompt):
            calls.append(prompt)
            return "[]"

    n = Narrator(backend=Counting(), batch_size=100)
    items = [{"record_id": f"b{i}", "residual_minor": 200, "amount_minor": 20_000,
              "causes": [("BANK_CHARGE", 0.98)]} for i in range(50)]
    n.narrate(items)
    n.narrate(items)
    assert len(calls) == 1, "the second pass must be served from cache"


def test_empty_input_makes_no_call():
    calls = []

    class Counting:
        def complete(self, prompt):
            calls.append(prompt); return "[]"

    assert Narrator(backend=Counting()).narrate([]) == []
    assert not calls


# --------------------------------------------------------------------------- #
# spending money must be deliberate
# --------------------------------------------------------------------------- #

def test_a_paid_model_is_refused_by_default():
    """Narration is an optional upgrade over templates that already work.
    A loop over a large batch is exactly where an accidental paid model gets
    expensive before anyone notices."""
    from allocation_agent.decide.openrouter import OpenRouterBackend, PaidModelRefused

    with pytest.raises(PaidModelRefused, match="free-tier"):
        OpenRouterBackend(api_key="k", model="openai/gpt-4o")


def test_a_free_model_is_allowed():
    from allocation_agent.decide.openrouter import OpenRouterBackend

    assert OpenRouterBackend(api_key="k", model="some/model:free").model.endswith(":free")


def test_paid_can_be_enabled_deliberately():
    from allocation_agent.decide.openrouter import OpenRouterBackend

    b = OpenRouterBackend(api_key="k", model="openai/gpt-4o", allow_paid=True)
    assert b.model == "openai/gpt-4o"


def test_the_refusal_says_how_to_override():
    from allocation_agent.decide.openrouter import OpenRouterBackend, PaidModelRefused

    with pytest.raises(PaidModelRefused, match="ALLOW_PAID_LLM"):
        OpenRouterBackend(api_key="k", model="anthropic/claude-3-opus")
