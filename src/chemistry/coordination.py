"""Deterministic coordination-geometry planning for Ln(III)-ligand complexes.

Single source of truth for the *chemistry* of a lanthanide extraction complex:
how big the coordination sphere is, which ligand atoms donate, how many ligand
copies fit, and what fills the rest of the inner sphere. ``build_dataset_no3d.py``
imports this module to write ``geometry_specs.csv``; ``build_unique_geometries.py``
then consumes those frozen specs for the 3D assembly.

Design choices (physically grounded, condition-aware where buildable):

* Coordination number is set by the *metal*, not by extraction conditions.
  Across La->Lu the ionic radius shrinks and the canonical aqua ion drops from
  the nonahydrate [Ln(H2O)9]3+ (tricapped trigonal prism, CN 9) to the
  octahydrate [Ln(H2O)8]3+ (square antiprism, CN 8). The inflection is the
  well-documented "gadolinium break". CN is therefore a function of Z, never a
  hardcoded constant.
* Denticity / donor set come from the ligand graph (RDKit). Amide/imide N is
  *not* a donor (its lone pair is delocalised into the C=O); the donor set of a
  diglycolamide is its carbonyl + ether oxygens.
* Stoichiometry (n_ligs) and the number of monodentate fill ligands follow from
  CN and denticity by mass balance.
* The acid is the condition that most directly reaches the first coordination
  sphere. Concentrated nitrate media give inner-sphere NO3-, dilute nitrate and
  non-nitrate media default to water. This sets the fill ligand only.

Everything here is deterministic: one (metal, ligand, acid context) -> one ComplexSpec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")


# ---------------------------------------------------------------------------
# Metal table: Z, lanthanide index, Shannon ionic radius (the contraction)
# ---------------------------------------------------------------------------
# Lanthanide(III) descriptors. `Ionic Radius_metal` is the Shannon effective
# ionic radius (CN=8, angstrom): the smooth La->Lu contraction is the single
# physical driver of adjacent-lanthanide selectivity, so it is the natural
# differential feature. `lanthanide_index` is 1..15 across La..Lu (Pm=5 absent
# from the dataset but kept for a contiguous index).
LANTHANIDE_DESCRIPTORS = {
    "La": {"Atomic Number_metal": 57, "lanthanide_index": 1, "Ionic Radius_metal": 1.160},
    "Ce": {"Atomic Number_metal": 58, "lanthanide_index": 2, "Ionic Radius_metal": 1.143},
    "Pr": {"Atomic Number_metal": 59, "lanthanide_index": 3, "Ionic Radius_metal": 1.126},
    "Nd": {"Atomic Number_metal": 60, "lanthanide_index": 4, "Ionic Radius_metal": 1.109},
    "Sm": {"Atomic Number_metal": 62, "lanthanide_index": 6, "Ionic Radius_metal": 1.079},
    "Eu": {"Atomic Number_metal": 63, "lanthanide_index": 7, "Ionic Radius_metal": 1.066},
    "Gd": {"Atomic Number_metal": 64, "lanthanide_index": 8, "Ionic Radius_metal": 1.053},
    "Tb": {"Atomic Number_metal": 65, "lanthanide_index": 9, "Ionic Radius_metal": 1.040},
    "Dy": {"Atomic Number_metal": 66, "lanthanide_index": 10, "Ionic Radius_metal": 1.027},
    "Ho": {"Atomic Number_metal": 67, "lanthanide_index": 11, "Ionic Radius_metal": 1.015},
    "Er": {"Atomic Number_metal": 68, "lanthanide_index": 12, "Ionic Radius_metal": 1.004},
    "Tm": {"Atomic Number_metal": 69, "lanthanide_index": 13, "Ionic Radius_metal": 0.994},
    "Yb": {"Atomic Number_metal": 70, "lanthanide_index": 14, "Ionic Radius_metal": 0.985},
    "Lu": {"Atomic Number_metal": 71, "lanthanide_index": 15, "Ionic Radius_metal": 0.977},
}

LANTHANIDES = list(LANTHANIDE_DESCRIPTORS.keys())
LANTHANIDE_SET = set(LANTHANIDES)

# Z -> element symbol for the lanthanides (+ Y, which tracks the heavy Ln).
_SYMBOL_BY_Z = {
    39: "Y", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm",
    63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm",
    70: "Yb", 71: "Lu",
}
_Z_BY_SYMBOL = {sym: z for z, sym in _SYMBOL_BY_Z.items()}


def symbol_for_Z(Z: int) -> str:
    try:
        return _SYMBOL_BY_Z.get(int(Z), f"Z{int(Z)}")
    except (TypeError, ValueError):
        return "?"


def z_for_symbol(symbol: str) -> Optional[int]:
    return _Z_BY_SYMBOL.get(str(symbol).strip())


# ---------------------------------------------------------------------------
# Coordination number from the metal (lanthanide contraction)
# ---------------------------------------------------------------------------
# La57..Gd64 -> CN 9   (light, [Ln(H2O)9]3+, tricapped trigonal prism)
# Tb65..Lu71 -> CN 8   (heavy, [Ln(H2O)8]3+, square antiprism)
# Y(39) tracks the late, smaller lanthanides (CN 8).
CN_BY_Z = {
    **{z: 9 for z in range(57, 65)},   # La..Gd
    **{z: 8 for z in range(65, 72)},   # Tb..Lu
    39: 8,                             # Y
}

DEFAULT_CN = 9


def cn_for_Z(Z: int, default: int = DEFAULT_CN) -> int:
    """Metal-dependent coordination number for Ln(III)."""
    try:
        return CN_BY_Z.get(int(Z), default)
    except (TypeError, ValueError):
        return default


# Shannon effective ionic radii for Ln(III), CN=8, in angstrom (mirrors the
# `Ionic Radius_metal` column above, keyed by Z for the geometry planner).
SHANNON_RADII_CN8 = {
    57: 1.160, 58: 1.143, 59: 1.126, 60: 1.109, 61: 1.093, 62: 1.079,
    63: 1.066, 64: 1.053, 65: 1.040, 66: 1.027, 67: 1.015, 68: 1.004,
    69: 0.994, 70: 0.985, 71: 0.977, 39: 1.019,  # Y(III) CN8 for reference
}


def ionic_radius_for_Z(Z: int) -> Optional[float]:
    """Shannon Ln(III) ionic radius (CN=8, angstrom); None if unknown."""
    try:
        return SHANNON_RADII_CN8.get(int(Z))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Donor detection from the ligand graph
# ---------------------------------------------------------------------------
# Oxophilic Ln(III) strongly prefers O and N donors; S/P appear in a few soft
# extractants. Carbon, halogens and metals are never donors here.
_DONOR_ELEMENTS = {"O", "N", "S", "P"}


def _bonded_to_chalcogenyl_carbon(
    atom: Chem.Atom,
    chalcogens: set[str] | frozenset[str] = frozenset({"O", "S"}),
) -> bool:
    """True if ``atom`` is bonded to a C=O/C=S centre.

    This covers ordinary amides as well as thioamides and aromatic imide
    nitrogens.  The latter were previously retained merely because RDKit marks
    them aromatic, even though their lone pair is delocalised into C=O.
    """
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "C":
            continue
        for b in nbr.GetBonds():
            other = b.GetOtherAtom(nbr)
            if (b.GetBondType() == Chem.BondType.DOUBLE
                    and other.GetSymbol() in chalcogens
                    and other.GetIdx() != atom.GetIdx()):
                return True
    return False


def _bonded_to_carbonyl_carbon(atom: Chem.Atom) -> bool:
    """Backward-compatible C=O-only helper used by older callers/tests."""
    return _bonded_to_chalcogenyl_carbon(atom, frozenset({"O"}))


def _is_carboxylic_acid_hydroxyl(atom: Chem.Atom) -> bool:
    """Whether O is the protonated O-H atom of a neutral carboxylic acid.

    A carboxylate contributes one coordination site in the mononuclear
    aminopolycarboxylate templates used here.  Counting both the carbonyl O and
    the neutral O-H from every COOH group inflated EDTA/CDTA-like ligands from
    their usual N2O4 hexadentate mode to an artificial octadentate template.
    """
    return (
        atom.GetSymbol() == "O"
        and atom.GetTotalNumHs() > 0
        and _bonded_to_carbonyl_carbon(atom)
    )


def _is_phosphorus_alkoxy_oxygen(atom: Chem.Atom) -> bool:
    """True for a neutral P-O-C ester oxygen, not P=O or P-OH."""
    if atom.GetSymbol() != "O":
        return False
    neighbor_symbols = {nbr.GetSymbol() for nbr in atom.GetNeighbors()}
    return "P" in neighbor_symbols and "C" in neighbor_symbols


def _is_candidate_donor(atom: Chem.Atom) -> bool:
    """A donor candidate is an O/N/S/P with a lone pair available to the metal.

    Exclusions, in order of chemical importance for these extractants:

    * cationic atoms (no lone pair to donate);
    * quaternary, H-free N/P (no lone pair);
    * amide, imide and thioamide nitrogens bonded to C=O/C=S;
    * pyrrolic aromatic N (degree 3; its lone pair is part of the aromatic
      sextet), while pyridine/phenanthroline N is retained;
    * the protonated O-H member of a neutral carboxylic acid pair;
    * neutral P-O-C ester oxygens, while phosphoryl P=O and P-OH are retained.
    """
    sym = atom.GetSymbol()
    if sym not in _DONOR_ELEMENTS:
        return False
    if atom.GetFormalCharge() > 0:
        return False
    if sym in {"N", "P"}:
        # Quaternary, H-free -> no lone pair.
        if atom.GetTotalDegree() >= 4 and atom.GetTotalNumHs() == 0 and atom.GetFormalCharge() == 0:
            return False
        # Amide/imide/thioamide N: deactivated by an adjacent C=O/C=S.  This
        # applies even when RDKit's ring model marks an imide N aromatic.
        if sym == "N" and _bonded_to_chalcogenyl_carbon(atom):
            return False
        # Pyrrolic or N-substituted azole N: no available pyridine-like lone
        # pair.  Aromatic donor N atoms have total degree 2.
        if sym == "N" and atom.GetIsAromatic() and atom.GetTotalDegree() >= 3:
            return False
    if _is_carboxylic_acid_hydroxyl(atom):
        return False
    if _is_phosphorus_alkoxy_oxygen(atom):
        return False
    return True


def _donor_label(mol: Chem.Mol, atom: Chem.Atom) -> str:
    """Human-readable donor label used for auditing the coordList."""
    sym = atom.GetSymbol()
    if sym == "O":
        # Distinguish carbonyl O (e.g. amide/ester) from ether/hydroxyl O.
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
            if nbr.GetSymbol() == "C" and bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                hetero = {n.GetSymbol() for n in nbr.GetNeighbors() if n.GetIdx() != atom.GetIdx()}
                if "N" in hetero:
                    return "O(amide_carbonyl)"
                if "O" in hetero:
                    return "O(ester_carbonyl)"
                return "O(carbonyl)"
        if atom.GetTotalNumHs() > 0:
            return "O(hydroxyl)"
        return "O(ether)"
    if sym == "N":
        if atom.GetIsAromatic():
            return "N(aromatic)"
        return "N(amine)"
    return f"{sym}(donor)"


@dataclass(frozen=True)
class DonorSet:
    """Donor atoms of a single ligand, as 0-based atom indices into its SMILES."""

    coord_list: List[int]
    donor_types: List[str]
    strategy: str = "graph_pocket"

    @property
    def denticity(self) -> int:
        return len(self.coord_list)


# Practical denticity ceiling: extractants in this dataset are 1-8 dentate.
MAX_DENTICITY = 8

# Two donors can chelate the *same* metal only if they sit close enough in the
# molecular graph to close a reasonable chelate ring. A chelate ring spanning
# donors that are `d` bonds apart has size d + 2 (the two metal-donor bonds), so
# a span of <= 5 bonds covers 4- to 7-membered rings -- the favourable range.
# Donors farther apart belong to *different* binding pockets (e.g. the separate
# DGA arms of a poly-DGA), which physically coordinate different metal centres.
MAX_CHELATE_SPAN = 5


def _polypyridyl_triazine_donors(
    mol: Chem.Mol,
    donors: List[int],
) -> Optional[List[int]]:
    """Recognise the BTP/BTBP/BTPhen/BTTP N-donor family.

    A 1,2,4-triazine ring contains three formally available aromatic N atoms,
    but BTP-family complexes bind through one N2 atom per terminal triazine,
    together with the pyridyl/phenanthroline N atoms of the central scaffold.
    Treating all triazine N atoms as simultaneous donors was the dominant
    systematic cause of N7/N8 regeneration failures.

    N2 is selected as the ring N that is both adjacent to another ring N and
    closest in the molecular graph to the central N-donor scaffold.  This is
    invariant to SMILES traversal direction and gives one donor per triazine.
    """
    triazine_rings: list[frozenset[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        if sum(atom.GetSymbol() == "N" for atom in atoms) != 3:
            continue
        candidate = frozenset(int(i) for i in ring)
        if candidate not in triazine_rings:
            triazine_rings.append(candidate)

    if not triazine_rings:
        return None

    triazine_atoms = set().union(*triazine_rings)
    central_n = [
        idx for idx in donors
        if idx not in triazine_atoms
        and mol.GetAtomWithIdx(idx).GetSymbol() == "N"
        and mol.GetAtomWithIdx(idx).GetIsAromatic()
    ]
    if not central_n:
        return None

    dist = Chem.GetDistanceMatrix(mol)
    selected = list(central_n)
    donor_set = set(donors)
    for ring in triazine_rings:
        ring_n = [
            idx for idx in ring
            if idx in donor_set and mol.GetAtomWithIdx(idx).GetSymbol() == "N"
        ]
        if not ring_n:
            return None
        n2_candidates = [
            idx for idx in ring_n
            if any(
                nbr.GetIdx() in ring and nbr.GetSymbol() == "N"
                for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            )
        ]
        pool = n2_candidates or ring_n
        selected.append(min(
            pool,
            key=lambda idx: (min(dist[idx][core] for core in central_n), idx),
        ))
    return sorted(set(selected))


def _pytri_donors(
    mol: Chem.Mol,
    donors: List[int],
) -> Optional[List[int]]:
    """Recognise 2,6-bis(1,2,3-triazolyl)pyridine N-donor pockets.

    PyTri ligands coordinate through the central pyridine N and the inward N3
    atom of each terminal triazole.  Counting both available N atoms of every
    triazole creates an impossible N5 pocket; the N1 atom bearing the alkyl
    substituent is already excluded by the generic donor filter.
    """
    triazole_rings: list[frozenset[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        if sum(atom.GetSymbol() == "N" for atom in atoms) != 3:
            continue
        candidate = frozenset(int(i) for i in ring)
        if candidate not in triazole_rings:
            triazole_rings.append(candidate)
    if len(triazole_rings) != 2:
        return None

    triazole_atoms = set().union(*triazole_rings)
    central_n = [
        idx for idx in donors
        if idx not in triazole_atoms
        and mol.GetAtomWithIdx(idx).GetSymbol() == "N"
        and mol.GetAtomWithIdx(idx).GetIsAromatic()
    ]
    if not central_n:
        return None

    dist = Chem.GetDistanceMatrix(mol)
    donor_set = set(donors)
    selected = list(central_n)
    for ring in triazole_rings:
        ring_n = [
            idx for idx in ring
            if idx in donor_set and mol.GetAtomWithIdx(idx).GetSymbol() == "N"
        ]
        if not ring_n:
            return None
        selected.append(min(
            ring_n,
            key=lambda idx: (min(dist[idx][core] for core in central_n), idx),
        ))
    return sorted(set(selected))


def _flavonol_oxo_hydroxyl_donors(
    mol: Chem.Mol,
    donors: List[int],
) -> Optional[List[int]]:
    """Return the primary 3-hydroxyl/4-carbonyl flavonol chelation site.

    Polyhydroxyflavonols such as quercetin expose many O atoms, but they are
    alternative local binding sites rather than one hepta-dentate pocket.  For
    the lanthanide extraction chemistry represented in this dataset the
    experimentally supported primary site is the adjacent 3-OH/4-oxo pair.
    """
    donor_set = set(donors)
    hydroxyl_oxygens = [
        idx for idx in donors
        if mol.GetAtomWithIdx(idx).GetSymbol() == "O"
        and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
    ]
    aromatic_ring_oxygen = any(
        atom.GetSymbol() == "O" and atom.GetIsAromatic()
        for atom in mol.GetAtoms()
    )
    # Require the polyhydroxy benzopyranone signature; do not reinterpret a
    # generic alpha-hydroxy ketone as a flavonol.
    if len(hydroxyl_oxygens) < 3 or not aromatic_ring_oxygen:
        return None

    candidate_pairs: list[tuple[int, int]] = []
    for oxygen_idx in donors:
        oxygen = mol.GetAtomWithIdx(oxygen_idx)
        if oxygen.GetSymbol() != "O" or oxygen.GetDegree() != 1:
            continue
        carbonyl_carbon = oxygen.GetNeighbors()[0]
        bond = mol.GetBondBetweenAtoms(oxygen_idx, carbonyl_carbon.GetIdx())
        if (carbonyl_carbon.GetSymbol() != "C"
                or bond.GetBondType() != Chem.BondType.DOUBLE):
            continue
        for adjacent_carbon in carbonyl_carbon.GetNeighbors():
            if adjacent_carbon.GetSymbol() != "C":
                continue
            for neighbor in adjacent_carbon.GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in donor_set and idx in hydroxyl_oxygens:
                    candidate_pairs.append((oxygen_idx, idx))
    if not candidate_pairs:
        return None
    return sorted(min(candidate_pairs))


def _compact_amide_core_donors(
    mol: Chem.Mol,
    donors: List[int],
    max_carbonyl_span: int = 8,
) -> Optional[List[int]]:
    """Return the donor atoms on one compact multi-amide chelation core.

    Terminal ether/methoxy groups on amide substituents are acceptors but are
    not part of the DGA/malonamide chelation pocket.  The intended pocket is
    the set of donor atoms lying on the shortest paths between two or more
    nearby amide-carbonyl oxygens (OO for diamides, OOO for DGAs, ONOOO for
    nitrilotriamides).  Widely separated multi-DGA arms fall back to the normal
    single-pocket selector.
    """
    amide_oxygens = [
        idx for idx in donors
        if mol.GetAtomWithIdx(idx).GetSymbol() == "O"
        and _donor_label(mol, mol.GetAtomWithIdx(idx)) == "O(amide_carbonyl)"
    ]
    if len(amide_oxygens) < 2:
        return None

    dist = Chem.GetDistanceMatrix(mol)
    max_span = max(
        dist[a][b]
        for pos, a in enumerate(amide_oxygens)
        for b in amide_oxygens[pos + 1:]
    )
    if max_span > int(max_carbonyl_span):
        return None

    path_atoms = set(amide_oxygens)
    for pos, a in enumerate(amide_oxygens):
        for b in amide_oxygens[pos + 1:]:
            path_atoms.update(Chem.rdmolops.GetShortestPath(mol, a, b))
    return sorted(idx for idx in donors if idx in path_atoms)


def _largest_pocket(mol: Chem.Mol, donors: List[int], max_span: int) -> List[int]:
    """Group donors into chelate pockets and return the largest one.

    Donors within `max_span` bonds of each other are joined (union-find); each
    connected component is one binding pocket. The largest component is the set
    that coordinates a single metal. Ties break toward the most compact pocket
    (smallest total pairwise distance), then lowest atom indices for determinism.
    """
    if len(donors) <= 1:
        return list(donors)

    dist = Chem.GetDistanceMatrix(mol)  # topological (bond-count) distances

    parent = {d: d for d in donors}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(donors):
        for b in donors[i + 1:]:
            if dist[a][b] <= max_span:
                union(a, b)

    pockets: dict[int, List[int]] = {}
    for d in donors:
        pockets.setdefault(find(d), []).append(d)

    def compactness(group: List[int]) -> float:
        if len(group) < 2:
            return 0.0
        return sum(dist[a][b] for i, a in enumerate(group) for b in group[i + 1:])

    best = min(
        pockets.values(),
        key=lambda g: (-len(g), compactness(g), min(g)),
    )
    return sorted(best)


def detect_donors(
    smiles: str,
    max_denticity: int = MAX_DENTICITY,
    single_pocket: bool = True,
    max_chelate_span: int = MAX_CHELATE_SPAN,
) -> Optional[DonorSet]:
    """Find the per-metal donor atoms of a ligand from its SMILES.

    Returns ``None`` if the SMILES does not parse or has no donor atoms. The
    indices are 0-based atom indices in RDKit's canonical ordering of the parsed
    molecule, so callers that need a stable SMILES/coordList pairing should pair
    them with ``Chem.MolToSmiles(mol)`` of the same parse.

    With ``single_pocket=True`` (default) the donors are reduced to the largest
    chelate pocket -- the set that can coordinate one metal. This is both the
    correct per-metal denticity and what assemblers like Architector expect: a
    multi-pocket ligand (e.g. a tris-DGA) coordinates one pocket per metal, not
    all its donors at once.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    all_donors: List[int] = []
    for atom in mol.GetAtoms():
        if _is_candidate_donor(atom):
            all_donors.append(atom.GetIdx())

    if not all_donors:
        return None

    strategy = "all_candidates"
    coord_list = list(all_donors)
    if single_pocket:
        family_donors = _polypyridyl_triazine_donors(mol, all_donors)
        if family_donors is not None:
            coord_list = family_donors
            strategy = "polypyridyl_triazine"
        else:
            pytri_donors = _pytri_donors(mol, all_donors)
            if pytri_donors is not None:
                coord_list = pytri_donors
                strategy = "pytri"
            else:
                flavonol_donors = _flavonol_oxo_hydroxyl_donors(mol, all_donors)
                if flavonol_donors is not None:
                    coord_list = flavonol_donors
                    strategy = "flavonol_3hydroxy_4oxo"
                else:
                    amide_core = _compact_amide_core_donors(mol, all_donors)
                    if amide_core is not None:
                        coord_list = amide_core
                        strategy = "compact_amide_core"
                    else:
                        coord_list = _largest_pocket(mol, all_donors, max_chelate_span)
                        strategy = "graph_pocket"

    if len(coord_list) > max_denticity:
        coord_list = coord_list[:max_denticity]

    donor_types = [_donor_label(mol, mol.GetAtomWithIdx(i)) for i in coord_list]
    return DonorSet(coord_list=coord_list, donor_types=donor_types, strategy=strategy)


def rank_donor_sets(
    smiles: str,
    max_candidates: int = 6,
    max_denticity: int = MAX_DENTICITY,
    max_chelate_span: int = MAX_CHELATE_SPAN,
) -> List[DonorSet]:
    """Return deterministic donor-set hypotheses without replacing the primary.

    ``detect_donors`` remains the source of the established family-aware first
    choice.  This function adds graph-derived alternatives for regeneration:
    donor neighbourhoods centred on each candidate atom and disconnected
    compatibility components.  The alternatives let QC distinguish a wrong
    coordination pocket from an unlucky placement instead of retrying the same
    frozen ``COORDLIST``.

    Candidates are ranked by denticity, maximum donor separation, total donor
    separation, then atom indices.  The validated primary is always first even
    when a graph candidate has a nominally better score.  No conformers or 3D
    coordinates are generated here; the calculation is purely topological.
    """
    limit = max(1, int(max_candidates))
    mol = Chem.MolFromSmiles(str(smiles))
    primary = detect_donors(
        smiles,
        max_denticity=max_denticity,
        single_pocket=True,
        max_chelate_span=max_chelate_span,
    )
    if mol is None or primary is None:
        return []

    donors = [atom.GetIdx() for atom in mol.GetAtoms() if _is_candidate_donor(atom)]
    if not donors:
        return [primary]
    dist = Chem.GetDistanceMatrix(mol)

    hypotheses: set[tuple[int, ...]] = {tuple(primary.coord_list)}

    # A centre-specific neighbourhood exposes overlapping pockets that the
    # transitive union used by _largest_pocket intentionally collapses.  This is
    # useful for ligands with several nearby but alternative binding sites.
    for centre in donors:
        pocket = tuple(sorted(
            donor for donor in donors
            if dist[centre][donor] <= int(max_chelate_span)
        )[:max_denticity])
        if pocket:
            hypotheses.add(pocket)

    # Preserve genuinely disconnected binding pockets as separate hypotheses.
    remaining = set(donors)
    while remaining:
        component: set[int] = set()
        frontier = [min(remaining)]
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(
                donor for donor in remaining
                if donor not in component
                and dist[current][donor] <= int(max_chelate_span)
            )
        remaining -= component
        hypotheses.add(tuple(sorted(component)[:max_denticity]))

    primary_key = tuple(primary.coord_list)

    def rank_key(indices: tuple[int, ...]) -> tuple:
        pair_distances = [
            float(dist[a][b])
            for pos, a in enumerate(indices)
            for b in indices[pos + 1:]
        ]
        return (
            -len(indices),
            max(pair_distances, default=0.0),
            sum(pair_distances),
            indices,
        )

    ordered = [primary_key] + sorted(
        (candidate for candidate in hypotheses if candidate != primary_key),
        key=rank_key,
    )
    result = [primary]
    for indices in ordered[1:limit]:
        result.append(DonorSet(
            coord_list=list(indices),
            donor_types=[
                _donor_label(mol, mol.GetAtomWithIdx(index)) for index in indices
            ],
            strategy="graph_pocket_ranked",
        ))
    return result


# ---------------------------------------------------------------------------
# Stoichiometry and fill
# ---------------------------------------------------------------------------

def choose_n_ligs(core_cn: int, denticity: int, max_ligs: int = 4) -> int:
    """How many copies of the ligand fit around the metal by mass balance."""
    dent = max(1, int(denticity))
    n = max(1, int(core_cn) // dent)
    return min(max_ligs, max(1, n))


def core_cn_for_donor_set(base_core_cn: int, donors: DonorSet) -> int:
    """Apply a family-imposed coordination number when it is unambiguous.

    BTP and PyTri are tridentate pincers that form three-ligand N9 spheres even
    for the smaller lanthanides.  The aqua-ion CN8/CN9 heuristic is therefore
    not the right constraint for these families.
    """
    if (donors.strategy in {"polypyridyl_triazine", "pytri"}
            and donors.denticity == 3):
        return 9
    return int(base_core_cn)


def choose_n_ligs_for_donor_set(
    core_cn: int,
    donors: DonorSet,
    ligand_smiles: str,
    max_ligs: int = 4,
) -> int:
    """Family-aware ligand stoichiometry after the donor template is known.

    * BTP (N3) uses three copies to form the established N9 complex.
    * PyTri (N3) likewise uses three copies for its dominant 1:3 complex.
    * The bulkier tetradentate/pentadentate BTBP, BTPhen and BTTP families use
      one ligand plus nitrate/water fill rather than two giant pincers.
    * A flavonol uses two bidentate 3-OH/4-oxo ligand copies.
    * Other very large (>=50 heavy atoms), >=4-dentate chelators use one copy;
      duplicating them is the common source of steric placement failures.
    """
    if donors.strategy in {"polypyridyl_triazine", "pytri"}:
        if donors.denticity == 3:
            return min(3, max(1, int(max_ligs)))
        if donors.denticity >= 4:
            return 1
    if donors.strategy == "flavonol_3hydroxy_4oxo" and donors.denticity == 2:
        return min(2, max(1, int(max_ligs)))
    try:
        mol = Chem.MolFromSmiles(str(ligand_smiles))
        if mol is not None and mol.GetNumHeavyAtoms() >= 50 and donors.denticity >= 4:
            return 1
    except Exception:
        pass
    return choose_n_ligs(core_cn, donors.denticity, max_ligs=max_ligs)


def n_fill(core_cn: int, denticity: int, n_ligs: int) -> int:
    """Number of monodentate fill ligands needed to saturate the metal."""
    return max(0, int(core_cn) - int(n_ligs) * int(denticity))


def complex_build_id(
    metal_Z: int,
    ligand_smiles: str,
    coord_list: List[int],
    denticity: int,
    core_cn: int,
    n_ligs: int,
    inner_sphere_anion: str,
    fill_ligand: str,
    n_fill_value: int,
) -> str:
    """Stable id for one physical complex, shared by planning and rescue."""
    coordlist_json = json.dumps([int(i) for i in coord_list])
    raw = "|".join([
        str(int(metal_Z)), str(ligand_smiles), coordlist_json, str(int(denticity)),
        str(int(core_cn)), str(int(n_ligs)), str(inner_sphere_anion),
        str(fill_ligand), str(int(n_fill_value)),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# The condition lever: which species fills the inner sphere
# ---------------------------------------------------------------------------
# Only the acid directly changes the buildable first-sphere fill species here.
# In concentrated nitrate media NO3- routinely coordinates inner-sphere (often
# bidentate); very dilute nitrate and non-nitrate media are treated as water-
# dominated. Temperature/contact time are not encoded as deterministic geometry
# changes; they belong in condition descriptors or a future conformer ensemble.
_NITRATE_TOKENS = ("hno3", "nitrate", "nitric")
NITRATE_INNER_SPHERE_MIN_M = 0.05


def inner_sphere_fill(acid_name: Optional[str], acid_concentration_M: Optional[float] = None) -> str:
    """Pick the inner-sphere fill ligand from the aqueous acid.

    Returns ``"nitrate"`` for nitrate media concentrated enough to plausibly
    populate the first sphere; otherwise returns ``"water"`` (the safe default
    for chloride/perchlorate/organic-acid/unknown/dilute-nitrate media).
    """
    if not acid_name:
        return "water"
    text = str(acid_name).strip().lower()
    if any(tok in text for tok in _NITRATE_TOKENS):
        if acid_concentration_M is None:
            return "nitrate"
        try:
            if float(acid_concentration_M) >= NITRATE_INNER_SPHERE_MIN_M:
                return "nitrate"
        except (TypeError, ValueError):
            return "nitrate"
    return "water"


# ---------------------------------------------------------------------------
# The single best complex spec for a row
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComplexSpec:
    """A deterministic, buildable description of one Ln(III)-ligand complex."""

    metal_Z: int
    metal_symbol: str
    metal_ox: int
    ligand_smiles: str
    core_cn: int
    denticity: int
    n_ligs: int
    coord_list: List[int]
    donor_types: List[str]
    fill_ligand: str          # "water" or "nitrate"
    n_fill: int
    valid: bool
    note: str = ""
    acid_name: Optional[str] = None
    extras: dict = field(default_factory=dict)


def plan_complex(
    metal_Z: int,
    ligand_smiles: str,
    acid_name: Optional[str] = None,
    metal_ox: int = 3,
    core_cn: Optional[int] = None,
    max_ligs: int = 4,
) -> ComplexSpec:
    """Plan the single best deterministic geometry for one (metal, ligand, acid).

    Parameters
    ----------
    metal_Z : atomic number of the lanthanide.
    ligand_smiles : extractant SMILES (already canonicalised by the caller).
    acid_name : aqueous acid (condition) -> sets the inner-sphere fill ligand.
    metal_ox : oxidation state (Ln(III) by default).
    core_cn : force a coordination number; ``None`` uses ``cn_for_Z`` (default).
    max_ligs : cap on ligand copies.
    """
    cn = int(core_cn) if core_cn is not None else cn_for_Z(metal_Z)
    symbol = symbol_for_Z(metal_Z)
    fill = inner_sphere_fill(acid_name)

    donors = detect_donors(ligand_smiles)
    if donors is None:
        return ComplexSpec(
            metal_Z=int(metal_Z), metal_symbol=symbol, metal_ox=int(metal_ox),
            ligand_smiles=str(ligand_smiles), core_cn=cn, denticity=0, n_ligs=0,
            coord_list=[], donor_types=[], fill_ligand=fill, n_fill=cn,
            valid=False, note="no_donors_or_unparseable_smiles",
            acid_name=acid_name,
        )

    if core_cn is None:
        cn = core_cn_for_donor_set(cn, donors)
    dent = donors.denticity
    n_ligs = choose_n_ligs_for_donor_set(
        cn, donors, ligand_smiles=ligand_smiles, max_ligs=max_ligs,
    )
    fill_count = n_fill(cn, dent, n_ligs)

    return ComplexSpec(
        metal_Z=int(metal_Z), metal_symbol=symbol, metal_ox=int(metal_ox),
        ligand_smiles=str(ligand_smiles), core_cn=cn, denticity=dent,
        n_ligs=n_ligs, coord_list=list(donors.coord_list),
        donor_types=list(donors.donor_types), fill_ligand=fill,
        n_fill=fill_count, valid=True,
        note="ok" if fill_count == 0 else f"fill_{fill_count}x_{fill}",
        acid_name=acid_name,
    )
