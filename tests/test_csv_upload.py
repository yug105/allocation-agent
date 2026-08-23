"""Bringing your own data.

A judge who cannot feed the system their own file has to take the demo's word
for everything. The demo runs on a dataset they have never seen, matched
against labels they cannot check; the only way to actually believe it is to put
in a file whose answers they already know.

So this path takes two CSVs -- a bank side and a ledger side -- and reconciles
them with the same code the demo runs. It is deliberately forgiving about
column names and utterly unforgiving about money.
"""

import pytest

from allocation_agent.adapters.csv_upload import (
    UploadError,
    parse_bank_csv,
    parse_ledger_csv,
    sniff_columns,
)

BANK = """id,account,amount,date
b1,ACC-1,1250.00,2026-03-01
b2,ACC-2,80.50,2026-03-02
"""

LEDGER = """key,account,amount,date
INV-1,ACC-1,1250.00,2026-03-01
INV-2,ACC-2,80.50,2026-03-03
"""


# --------------------------------------------------------------------------- #
# column discovery -- nobody's export is named the way ours is
# --------------------------------------------------------------------------- #

def test_finds_columns_under_their_own_names():
    cols = sniff_columns(["Txn Amt (INR)", "Val Dt", "A/c No", "Narration"])
    assert cols["amount"] == "Txn Amt (INR)"
    assert cols["date"] == "Val Dt"
    assert cols["account"] == "A/c No"


def test_exact_names_win_over_fuzzy_ones():
    cols = sniff_columns(["amount", "settled_amount", "date", "account"])
    assert cols["amount"] == "amount"


def test_a_missing_column_is_named_in_the_error_not_guessed():
    with pytest.raises(UploadError, match="date"):
        parse_bank_csv("id,account,amount\nb1,A,1.00\n")


# --------------------------------------------------------------------------- #
# money -- the rule that must not bend
# --------------------------------------------------------------------------- #

def test_amounts_become_integer_minor_units():
    recs = parse_bank_csv(BANK)
    assert recs[0].amount_minor == 125_000
    assert recs[1].amount_minor == 8_050


def test_a_third_decimal_is_refused_rather_than_rounded():
    """Rounding someone's money silently is the one thing never to do."""
    with pytest.raises(UploadError, match="precision"):
        parse_bank_csv("id,account,amount,date\nb1,A,1.005,2026-03-01\n")


def test_a_row_with_unreadable_money_names_its_line_number():
    with pytest.raises(UploadError, match="row 3"):
        parse_bank_csv("id,account,amount,date\nb1,A,1.00,2026-03-01\nb2,A,abc,2026-03-01\n")


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stamp", ["2026-03-01", "01/03/2026", "03/01/2026", "1 Mar 2026"])
def test_common_date_layouts_are_accepted(stamp):
    recs = parse_bank_csv(f"id,account,amount,date\nb1,A,1.00,{stamp}\n")
    assert recs[0].day is not None


def test_one_layout_is_chosen_for_the_whole_file_not_per_row():
    """03/01 and 05/02 are both readable two ways. Deciding per row silently
    mixes day-first and month-first inside one file."""
    text = "id,account,amount,date\n" + "".join(
        f"b{i},A,1.00,{d}\n" for i, d in enumerate(["13/01/2026", "05/02/2026"]))
    recs = parse_bank_csv(text)
    assert recs[1].day - recs[0].day == 23        # 13 Jan -> 5 Feb, day-first


def test_an_unreadable_date_is_refused():
    with pytest.raises(UploadError, match="date"):
        parse_bank_csv("id,account,amount,date\nb1,A,1.00,not-a-date\n")


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #

def test_ledger_rows_carry_the_key_they_belong_to():
    rows = parse_ledger_csv(LEDGER)
    assert rows[0].key == "INV-1"
    assert rows[0].account == "ACC-1"


def test_a_ledger_without_a_key_column_uses_the_row_number():
    """An export with no invoice column still reconciles; the key is positional
    and the user can see which row it was."""
    rows = parse_ledger_csv("account,amount,date\nACC-1,5.00,2026-03-01\n")
    assert rows[0].key == "row-2"


def test_bank_rows_without_an_id_column_are_numbered():
    recs = parse_bank_csv("account,amount,date\nACC-1,5.00,2026-03-01\n")
    assert recs[0].record_id == "row-2"


def test_an_empty_file_is_refused_with_a_readable_message():
    with pytest.raises(UploadError, match="no rows"):
        parse_bank_csv("id,account,amount,date\n")


def test_blank_account_is_kept_as_absent_not_as_empty_string():
    recs = parse_bank_csv("id,account,amount,date\nb1,,5.00,2026-03-01\n")
    assert recs[0].account is None


def test_the_two_sides_line_up_on_the_same_day_scale():
    """A bank row and a ledger row on the same calendar date must produce the
    same day number, or every date feature is quietly wrong."""
    recs = parse_bank_csv(BANK)
    rows = parse_ledger_csv(LEDGER)
    assert recs[0].day == rows[0].day
