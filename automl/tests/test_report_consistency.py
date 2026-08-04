#!/usr/bin/env python3
"""Every headline number in a report must appear in the CSV it came from.

Why this exists
---------------
This study's worst bug was a figure and a table computing the same quantity two
different ways, and its most persistent one is prose drifting from its own
data. `PI_SWEEP_RESULTS.md`'s summary block still asserted a "+0.064 gap" that
its own §9 had withdrawn; `SYNTHESIS.md` repeated it. Nobody noticed for a week,
because nothing checked.

So: for each claim below, read the number out of the result CSV and assert the
string appears in every document that quotes it. It cannot catch a wrong number
that is quoted consistently, but it catches the failure mode this project
actually has -- a corrected table and an uncorrected paragraph.

Skips cleanly when the CSVs are absent, so a fresh checkout without the
artifacts still passes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# (label, csv, pandas query, column, documents that quote it)
CLAIMS = [
    ("dual-key drop-in, binned", "dualkey_test.csv",
     "key=='composition_key' and base=='no topology (CatBoost+repaired)'",
     "delta",
     ["DUALKEY_RESULTS.md", "REANALYSIS_2026-07-29.md", "SYNTHESIS.md",
      "README.md"]),
    ("dual-key drop-in, strict", "dualkey_test.csv",
     "key=='strict_composition_key' and base=='no topology (CatBoost+repaired)'",
     "delta",
     ["DUALKEY_RESULTS.md", "REANALYSIS_2026-07-29.md"]),
    ("encoder decisive contrast", "encoder_test.csv",
     "key=='composition_key' and base=='with D0'", "delta",
     ["ENCODER_RESULTS.md", "REANALYSIS_2026-07-29.md", "SYNTHESIS.md"]),
    ("full stack over all arms", "full_stack.csv",
     "key=='composition_key'", "adj_r2",
     ["REANALYSIS_2026-07-29.md"]),
    ("objective strict, adds to no-topology", "objective_test.csv",
     "key=='strict_composition_key' and base=='no topology'", "delta",
     ["OBJECTIVE_RESULTS.md", "REANALYSIS_2026-07-29.md"]),
    ("objective binned, vs S0", "objective_test.csv",
     "key=='composition_key' and base=='with S0'", "delta",
     ["OBJECTIVE_RESULTS.md", "REANALYSIS_2026-07-29.md"]),
    # SWEEP2.  The confirmatory delta is the number the whole campaign turns
    # on, and A1's collapse is its one large effect; both must stay tied to
    # the CSVs that produced them.
    ("sweep2 confirmatory, binned", "sweep2_test.csv",
     "key=='composition_key' and cell=='C1'", "delta",
     ["SWEEP2_RESULTS.md"]),
    ("sweep2 A1 collapse", "sweep2_cells.csv", "cell=='A1'", "gain_vs_A0",
     ["SWEEP2_RESULTS.md"]),
    ("sweep2 C1 screen gain", "sweep2_cells.csv", "cell=='C1'", "gain_vs_A0",
     ["SWEEP2_RESULTS.md"]),
    ("campaign3 T2 gain", "c3_cells.csv", "cell=='T2'", "gain_vs_D0",
     ["CAMPAIGN3_RESULTS.md"]),
    ("campaign3 T2X gain", "c3_cells.csv", "cell=='T2X'", "gain_vs_D0",
     ["CAMPAIGN3_RESULTS.md"]),
    ("campaign3 T3 gain", "c3_cells.csv", "cell=='T3'", "gain_vs_D0",
     ["CAMPAIGN3_RESULTS.md"]),
    ("campaign4 neutral gain", "c4_cells.csv", "cell=='neutral'",
     "gain_vs_control", ["CAMPAIGN4_RESULTS.md"]),
    ("campaign4 shipped gain", "c4_cells.csv", "cell=='shipped'",
     "gain_vs_control", ["CAMPAIGN4_RESULTS.md"]),
]


def _forms(v: float) -> list[str]:
    """Renderings a report might legitimately use for one number."""
    return [f"{v:+.4f}", f"{v:.4f}", f"{abs(v):.4f}", f"{v:+.3f}", f"{v:.3f}"]


@pytest.mark.parametrize("label,csv,query,col,docs", CLAIMS,
                         ids=[c[0] for c in CLAIMS])
def test_reports_quote_their_own_csv(label, csv, query, col, docs):
    path = REPORTS / csv
    if not path.exists():
        pytest.skip(f"{csv} not present (artifacts not built)")
    frame = pd.read_csv(path).query(query)
    if frame.empty:
        pytest.skip(f"{csv} has no row matching {query!r}")
    value = float(frame[col].iloc[0])
    forms = _forms(value)
    for doc in docs:
        p = REPORTS / doc
        if not p.exists():
            pytest.skip(f"{doc} not present")
        text = p.read_text()
        assert any(f in text for f in forms), (
            f"{doc} quotes no rendering of {label} = {value:+.4f} "
            f"(tried {forms}); the document has drifted from {csv}")


def test_sweep2_confirmatory_verdict_is_not_overstated():
    """C1 did not replicate; the report must not claim it did.

    The screen's winner beat its gate on the tune half and then failed the
    single confirmatory look after correction.  That is the outcome most likely
    to drift in prose -- an uncorrected interval excluding zero is right there
    in the CSV, and quoting it without the correction would turn a null into a
    finding.
    """
    path = REPORTS / "sweep2_test.csv"
    if not path.exists():
        pytest.skip("sweep2_test.csv not present")
    d = pd.read_csv(path)
    binned = d[d["key"] == "composition_key"]
    if binned.empty:
        pytest.skip("no binned row")
    assert (binned["verdict_corrected"] == "not distinguishable").all(), (
        "sweep2_test.csv now shows a corrected verdict other than 'not "
        "distinguishable'; SWEEP2_RESULTS.md asserts the sweep is a null and "
        "must be revised if that changed")
    assert (binned["n_seeds"] == 16).all(), (
        "the confirmatory contrast must be computed at 16 seeds a side")
    text = (REPORTS / "SWEEP2_RESULTS.md").read_text()
    assert "did not replicate" in text


def test_ceiling_is_withdrawn():
    """Every ceiling_test estimator must be marked invalid, with a reason.

    Withdrawn 30 July 2026 (audit E1): E2 measured a quantity the model predicts
    and the metric averages out; E1/E3 measured a non-representative subset.  This
    pins the withdrawal so a future edit cannot quietly restore a number.
    """
    path = REPORTS / "ceiling_test.csv"
    if not path.exists():
        pytest.skip("ceiling_test.csv not present")
    d = pd.read_csv(path)
    assert "valid" in d.columns, "ceiling_test.csv must carry a validity column"
    assert not d["valid"].any(), (
        "ceiling_test marked an estimator valid; all three were withdrawn "
        "(AUDIT_2026-07-30.md E1)")
    assert "withdrawn_reason" in d.columns and d["withdrawn_reason"].notna().all()


def test_ceiling_v2_reports_identifiability():
    """The replacement must say whether a ceiling is identifiable at all."""
    path = REPORTS / "ceiling_v2.csv"
    if not path.exists():
        pytest.skip("ceiling_v2.csv not present")
    d = pd.read_csv(path).set_index("quantity")
    assert "implied ceiling if sigma transfers" in d.index
    # 94% within one DOI is the finding that refuted the source-conflict premise
    assert float(d.loc["frac within one DOI", "value"]) > 0.5


def test_published_headline_still_matches_stack_test():
    """The three published contrasts must not move.

    stack_test.csv is regenerated by several commands.  It was once overwritten
    at n_boot=100 by a verification job, which silently degraded it; this pins
    both the values and the draw count.
    """
    path = REPORTS / "stack_test.csv"
    if not path.exists():
        pytest.skip("stack_test.csv not present")
    d = pd.read_csv(path).set_index("contrast")
    assert (d["n_boot"] == 400).all(), (
        "stack_test.csv was written with the wrong number of bootstrap draws; "
        "re-run `python3 -m automl.topo.stack_test --n-boot 400`")
    assert d.loc["1_primary", "delta"] == pytest.approx(0.0351, abs=5e-4)
    assert d.loc["3_decisive", "delta"] == pytest.approx(0.0296, abs=5e-4)


# ---------------------------------------------------------------------------
# Withdrawn numbers.  A claim that has been retracted must not survive anywhere
# except in the documents that retract it.  This is the guard that would have
# caught PI_SWEEP_RESULTS asserting a "+0.064 gap" its own section 9 withdrew.
WITHDRAWN = [
    ("+0.679", "the ceiling estimate withdrawn 30 July 2026 (audit E1)"),
    ("+0.412", "the 'headroom' derived from that ceiling"),
    ("39 %", "'39% of attainable', derived from that ceiling"),
    ("39% of", "'39% of attainable', derived from that ceiling"),
]
# Documents whose job is to discuss the withdrawal.
ALLOWED = {"AUDIT_2026-07-30.md", "ceiling_test.csv", "ceiling_v2.csv"}


def _is_marked(lines: list[str], i: int) -> bool:
    """Is the occurrence on line ``i`` explicitly marked as withdrawn?

    Line-aware rather than document-aware, because this project's convention is
    to strike a retracted claim through **in place** and leave it visible.  So an
    occurrence is acceptable if it is struck through, or sits on/near a
    withdrawal marker, or falls after the document's erratum heading.
    """
    markers = ("WITHDRAWN", "Withdrawn", "withdrawn", "Erratum", "erratum",
               "Correction,", "~~")
    window = " ".join(lines[max(0, i - 2): i + 3])
    return any(m in window for m in markers)


def _offending_lines(text: str, token: str) -> list[int]:
    """1-indexed lines asserting ``token`` without a withdrawal marker."""
    lines = text.splitlines()
    # everything after an erratum/withdrawal heading is discussion of the
    # retraction and is allowed to quote the number
    cut = len(lines)
    for i, ln in enumerate(lines):
        if ln.startswith("##") and any(
                m in ln for m in ("Erratum", "erratum", "WITHDRAWN",
                                  "Withdrawn", "withdrawn", "Correction,")):
            cut = i
            break
    return [i + 1 for i in range(cut)
            if token in lines[i] and not _is_marked(lines, i)]


@pytest.mark.parametrize("token,why", WITHDRAWN, ids=[w[0] for w in WITHDRAWN])
def test_withdrawn_numbers_are_not_asserted(token, why):
    offenders = {}
    for p in sorted(REPORTS.glob("*.md")):
        if p.name in ALLOWED:
            continue
        bad = _offending_lines(p.read_text(), token)
        if bad:
            offenders[p.name] = bad
    assert not offenders, (
        f"{token} ({why}) is asserted without a withdrawal marker at "
        f"{offenders}. Strike it through in place, or move it below the "
        f"erratum heading.")


def test_campaign3_reports_a_null_and_spends_no_confirmatory_look():
    """No cell cleared the gate, so no confirmatory contrast may exist.

    The screen's best cell is NEGATIVE (-0.0253).  If a c3_test.csv ever appears
    it means a confirmatory look was spent on a cell that did not earn one, and
    the report's "no confirmatory run was spent" claim -- plus the intact
    26-look budget it implies -- would be false.
    """
    cells = REPORTS / "c3_cells.csv"
    if not cells.exists():
        pytest.skip("c3_cells.csv not present")
    d = pd.read_csv(cells)
    best = d[d["cell"] != "D0"]["gain_vs_D0"].max()
    assert best <= 0.005, (
        f"a campaign-3 cell now clears the +0.005 gate at {best:+.4f}; "
        f"CAMPAIGN3_RESULTS.md asserts a null and must be revised")
    assert not (REPORTS / "c3_test.csv").exists(), (
        "c3_test.csv exists, so a confirmatory look was spent -- but no cell "
        "cleared the screening gate that authorises one")


def test_campaign4_reports_a_null_and_spends_no_confirmatory_look():
    """No arm cleared the gate, so no confirmatory contrast may exist.

    The strict-key value (+0.0188) is larger than the gated binned one
    (+0.0040) and is exactly the number that would be tempting to promote after
    the fact.  This pins that it was not: if c4_test.csv ever appears, a
    confirmatory look was spent on an arm that did not earn one, and the
    report's null plus its intact 29-look budget become false.
    """
    cells = REPORTS / "c4_cells.csv"
    if not cells.exists():
        pytest.skip("c4_cells.csv not present")
    d = pd.read_csv(cells)
    best = d[d["cell"] != "control"]["gain_vs_control"].max()
    assert best <= 0.005, (
        f"a campaign-4 arm now clears the +0.005 gate at {best:+.4f}; "
        f"CAMPAIGN4_RESULTS.md asserts a null and must be revised")
    assert not (REPORTS / "c4_test.csv").exists(), (
        "c4_test.csv exists, so a confirmatory look was spent -- but no arm "
        "cleared the screening gate that authorises one")


def test_campaign4_arms_share_one_row_set():
    """All three arms must be scored on the same complexes.

    The shipped arm once loaded the 956-complex published asset while the others
    loaded the 627-complex subset, which made every cross-arm contrast a
    comparison of two different datasets while looking entirely normal.
    """
    cells = REPORTS / "c4_cells.csv"
    if not cells.exists():
        pytest.skip("c4_cells.csv not present")
    d = pd.read_csv(cells)
    assert set(d["cell"]) == {"shipped", "control", "neutral"}
    assert (d["n_seeds"] == 4).all(), "every arm needs all four seeds"


def test_campaign3_anchor_reproduces_the_sweep2_anchor():
    """D0 and A0 are the same configuration and must give the same number.

    Campaign 3 keeps the sweep2 anchor unchanged precisely so the two campaigns
    are comparable.  Determinism makes that checkable rather than assumed: if
    these ever diverge, something in the shared training path moved and every
    cross-campaign comparison in the reports is void.
    """
    a = REPORTS / "sweep2_cells.csv"
    b = REPORTS / "c3_cells.csv"
    if not (a.exists() and b.exists()):
        pytest.skip("campaign CSVs not both present")
    s2 = pd.read_csv(a).query("cell=='A0'")
    c3 = pd.read_csv(b).query("cell=='D0'")
    if s2.empty or c3.empty:
        pytest.skip("anchor row missing")
    for col in ("tune_adj_binned", "tune_adj_strict", "tune_r2_overall"):
        assert float(s2[col].iloc[0]) == pytest.approx(
            float(c3[col].iloc[0]), abs=1e-6), (
            f"the campaign-3 anchor no longer reproduces the sweep2 anchor on "
            f"{col}; cross-campaign comparisons are void until this is resolved")
