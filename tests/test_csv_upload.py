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


# --------------------------------------------------------------------------- #
# What Excel actually exports.
#
# A European Excel writes `a;b;c` with a comma for the decimal point. That file
# was refused with "could not find a column for: account", because the whole
# row parsed as one column. Supporting it means supporting the decimal comma in
# the same breath: read `1250,00` with a comma-stripping parser and you get
# 125,000.00 -- a hundredfold error, silent, in money.
# --------------------------------------------------------------------------- #

def test_a_semicolon_file_is_read_rather_than_refused():
    recs = parse_bank_csv("id;account;amount;date\nB1;ACC;1250,00;2026-03-01\n")
    assert len(recs) == 1
    assert recs[0].account == "ACC"


def test_the_decimal_comma_travels_with_the_semicolon():
    """1250,00 is one thousand two hundred and fifty, not a hundred and
    twenty-five thousand."""
    recs = parse_bank_csv("id;account;amount;date\nB1;ACC;1250,00;2026-03-01\n")
    assert recs[0].amount_minor == 125_000


def test_a_comma_file_still_reads_commas_as_thousands():
    recs = parse_bank_csv('id,account,amount,date\nB1,ACC,"1,250.00",2026-03-01\n')
    assert recs[0].amount_minor == 125_000


def test_a_tab_separated_export_is_read():
    recs = parse_bank_csv("id\taccount\tamount\tdate\nB1\tACC\t1250.00\t2026-03-01\n")
    assert recs[0].amount_minor == 125_000


def test_a_semicolon_file_still_refuses_excess_precision():
    with pytest.raises(UploadError, match="precision"):
        parse_bank_csv("id;account;amount;date\nB1;ACC;1,005;2026-03-01\n")


# --------------------------------------------------------------------------- #
# 03/01/2026 is the third of January or the first of March depending on which
# side of the Atlantic wrote it. One is picked for the whole file; a caller who
# is never told which has no way to notice the wrong one.
# --------------------------------------------------------------------------- #

def test_the_chosen_date_layout_is_reported():
    recs, layout = parse_bank_csv("id,account,amount,date\nB1,A,1.00,03/01/2026\n",
                                  report_layout=True)
    assert layout in {"DD/MM/YYYY", "MM/DD/YYYY"}
    assert recs[0].day is not None


def test_an_unambiguous_file_reports_the_layout_it_used():
    _, layout = parse_bank_csv("id,account,amount,date\nB1,A,1.00,2026-03-01\n",
                               report_layout=True)
    assert layout == "YYYY-MM-DD"


# --------------------------------------------------------------------------- #
# The message a person has to act on.
# --------------------------------------------------------------------------- #

def test_a_non_numeric_amount_is_named_not_called_blank():
    """'N/A' reported "the amount is blank", sending someone to look for an
    empty cell that is not empty."""
    with pytest.raises(UploadError, match="N/A"):
        parse_bank_csv("id,account,amount,date\nB1,A,N/A,2026-03-01\n")


def test_a_genuinely_blank_amount_still_says_blank():
    with pytest.raises(UploadError, match="blank"):
        parse_bank_csv("id,account,amount,date\nB1,A,,2026-03-01\n")


def test_an_entirely_empty_file_is_refused_not_crashed():
    """Sniffing the delimiter reads the first row; on an empty file there is
    none, and `next()` without a default raised StopIteration as a 500."""
    with pytest.raises(UploadError, match="header"):
        parse_bank_csv("")


def test_a_file_of_only_newlines_is_refused():
    with pytest.raises(UploadError):
        parse_bank_csv("\n\n\n")
