"""Regression tests for chemically distinct ligand-family donor templates.

Runnable with pytest, or standalone:
``.venv/bin/python tests/test_coordination_families.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chemistry.coordination import (  # noqa: E402
    detect_donors,
    plan_complex,
    rank_donor_sets,
)


def _donors(smiles: str):
    result = detect_donors(smiles)
    assert result is not None
    return result


def test_btp_is_tridentate_not_all_seven_aromatic_nitrogens():
    smiles = "CCCc1nnc(-c2cccc(-c3nnc(CCC)c(CCC)n3)n2)nc1CCC"
    result = _donors(smiles)
    assert result.coord_list == [5, 13, 24]
    assert result.donor_types == ["N(aromatic)"] * 3
    assert result.strategy == "polypyridyl_triazine"

    # Yb would normally inherit CN8 from the aqua-ion heuristic.  BTP instead
    # needs the family-specific Ln(BTP)3 N9 template.
    spec = plan_complex(70, smiles, acid_name="HNO3")
    assert spec.core_cn == 9
    assert spec.n_ligs == 3
    assert spec.n_fill == 0


def test_btbp_is_tetradentate_not_all_eight_aromatic_nitrogens():
    smiles = (
        "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC"
    )
    result = _donors(smiles)
    assert result.coord_list == [7, 20, 35, 36]
    assert result.donor_types == ["N(aromatic)"] * 4
    assert result.strategy == "polypyridyl_triazine"

    # A single bulky BTBP chelator plus inner-sphere fill is safer and matches
    # the observed 1:1 nitrate-family motif; two ligand copies clash heavily.
    spec = plan_complex(63, smiles, acid_name="HNO3")
    assert spec.core_cn == 9
    assert spec.n_ligs == 1
    assert spec.n_fill == 5


def test_espytri_uses_inward_triazole_n_and_three_ligand_n9_stoichiometry():
    smiles = "CCCCCCCCn1cc(-c2cc(CC(=O)O)cc(-c3cn(CCCCCCCC)nn3)n2)nn1"
    result = _donors(smiles)
    assert result.coord_list == [32, 33, 34]
    assert result.donor_types == ["N(aromatic)"] * 3
    assert result.strategy == "pytri"
    spec = plan_complex(63, smiles, acid_name="HNO3")
    assert spec.core_cn == 9
    assert spec.n_ligs == 3
    assert spec.n_fill == 0


def test_quercetin_uses_local_3_hydroxyl_4_oxo_site_and_two_ligands():
    smiles = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
    result = _donors(smiles)
    assert result.coord_list == [0, 3]
    assert result.donor_types == ["O(carbonyl)", "O(hydroxyl)"]
    assert result.strategy == "flavonol_3hydroxy_4oxo"
    spec = plan_complex(63, smiles, acid_name="HNO3")
    assert spec.core_cn == 9
    assert spec.n_ligs == 2
    assert spec.n_fill == 5


def test_hedta_uses_two_n_three_carboxylate_o_and_terminal_alcohol_o():
    result = _donors("O=C(O)CN(CCO)CCN(CC(=O)O)CC(=O)O")
    assert result.coord_list == [0, 4, 7, 10, 13, 17]
    assert result.donor_types.count("N(amine)") == 2
    assert result.donor_types.count("O(ester_carbonyl)") == 3
    assert result.donor_types.count("O(hydroxyl)") == 1


def test_cdta_uses_one_oxygen_per_carboxylate_plus_two_amines():
    result = _donors("O=C(O)CN(CC(=O)O)C1CCCCC1N(CC(=O)O)CC(=O)O")
    assert result.coord_list == [0, 4, 7, 15, 18, 22]
    assert result.donor_types.count("N(amine)") == 2
    assert result.donor_types.count("O(ester_carbonyl)") == 4
    assert "O(hydroxyl)" not in result.donor_types


def test_terminal_methoxy_arms_do_not_expand_dga_chelation_core():
    result = _donors("COCCN(CCOC)C(=O)COCC(=O)N(CCOC)CCOC")
    assert result.coord_list == [10, 12, 15]
    assert result.donor_types == [
        "O(amide_carbonyl)", "O(ether)", "O(amide_carbonyl)",
    ]
    assert result.strategy == "compact_amide_core"


def test_aromatic_imide_nitrogens_are_not_counted_as_pbi_donors():
    smiles = (
        "CCCCCCn1c(-c2ccccc2)c(-c2ccccc2)c2cc3ccc4cc5c(-c6ccccc6)"
        "c(-c6ccccc6)n(CCCCCC)c(=O)c5nc4c3nc2c1=O"
    )
    result = _donors(smiles)
    assert result.coord_list == [51, 53, 56, 59]
    assert result.donor_types == [
        "O(amide_carbonyl)", "N(aromatic)", "N(aromatic)", "O(amide_carbonyl)",
    ]
    spec = plan_complex(63, smiles, acid_name="HNO3")
    assert spec.n_ligs == 1
    assert spec.n_fill == 5


def test_phosphoryl_phenanthroline_excludes_p_o_ethyl_oxygens():
    result = _donors("CCOP(=O)(OCC)c1ccc2ccc3ccc(P(=O)(OCC)OCC)nc3c2n1")
    assert result.denticity == 4
    assert result.coord_list == [4, 19, 26, 29]
    assert result.donor_types.count("N(aromatic)") == 2


def test_ranked_donor_sets_preserve_primary_and_expose_overlapping_pockets():
    # The terminal oxygens are six bonds apart, but each is a plausible local
    # O,O pocket with the central oxygen. The existing primary is preserved;
    # regeneration can explicitly evaluate the two alternatives after QC.
    smiles = "OCCOCCO"
    primary = _donors(smiles)
    ranked = rank_donor_sets(smiles)

    assert ranked[0] == primary
    assert [candidate.coord_list for candidate in ranked] == [
        [0, 3, 6], [0, 3], [3, 6],
    ]
    assert [candidate.donor_types for candidate in ranked] == [
        ["O(hydroxyl)", "O(ether)", "O(hydroxyl)"],
        ["O(hydroxyl)", "O(ether)"],
        ["O(ether)", "O(hydroxyl)"],
    ]
    assert [candidate.strategy for candidate in ranked[1:]] == [
        "graph_pocket_ranked", "graph_pocket_ranked",
    ]


def test_ranked_donor_sets_are_deterministic_and_bounded():
    smiles = "OCCOCCO"
    first = rank_donor_sets(smiles, max_candidates=2)
    second = rank_donor_sets(smiles, max_candidates=2)
    assert first == second
    assert len(first) == 2


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
