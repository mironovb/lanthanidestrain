#!/usr/bin/env python3
"""Ln-xTB (Zhang 2026, J. Comput. Chem., 10.1002/jcc.70321) as a drop-in
parameter patch for the stock xtb binary.

Ln-xTB re-optimises the 20 element-specific GFN2-xTB parameters of La-Lu
(same 5d6s6p valence basis, f in core) against all-electron Ln-X bond lengths
in LnX3 / Ln(H2O)9 / Ln(C301)3 / Ln(P507)3 / Ln(POO)3.  The SI ships the
values twice -- Tables S60/S61 (element columns) and Table S62 (xtb-format
$Z blocks) -- and only as PDF text, no parameter file.  This module

  1. parses both copies from the SI text and cross-checks them,
  2. splices them into a copy of xtb's param_gfn2-xtb.txt (Z = 57..71 blocks
     replaced, everything else untouched) under automl/artifacts/lnxtb/,
  3. writes lnxtb_params.json alongside, so the artefact is reproducible
     without the PDF.

Running xtb with XTBPATH pointing at that directory and ``--gfn 2`` is then
Ln-xTB.  Verified by reproducing the SI's optimised-structure header energies
to 1e-6 Eh (see ``--verify``).

    pdftotext jcc70321-sup-0002-datas2.pdf si.txt   # or pypdf
    python3 -m automl.qc.lnxtb_params --si-text si.txt --verify <xyz dir>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "automl/artifacts/lnxtb"
STOCK_HOME = Path.home() / "opt/xtb-6.7.1"
KEYS = ["lev1", "lev2", "lev3", "exp1", "exp2", "exp3", "GAM", "GAM3",
        "KCNS", "KCNP", "KCND", "DPOL", "QPOL", "REPA", "REPB",
        "POLYS", "POLYP", "POLYD", "LPARP", "LPARD"]
LN = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def _num(tok: str) -> float:
    return float(tok.replace("–", "-").replace("−", "-"))


def parse_tables(text: str) -> dict[str, dict[str, float]]:
    """Tables S60 (La-Gd) and S61 (Tb-Lu): one row per parameter."""
    out: dict[str, dict[str, float]] = {m: {} for m in LN}
    for hdr, elems in (("Table S60", LN[:8]), ("Table S61", LN[8:])):
        i = re.search(hdr.replace(" ", r"\s*"), text).start()
        block = text[i:i + 4000]
        for k in KEYS:
            m = re.search(rf"^{k}\s+(.+)$", block, re.M)
            vals = [_num(t) for t in m.group(1).split()]
            assert len(vals) == len(elems), (hdr, k, vals)
            for e, v in zip(elems, vals):
                out[e][k] = v
    return out


def parse_blocks(text: str) -> dict[str, dict[str, float]]:
    """Table S62: $Z=57 .. $Z=71 blocks (skip the Ce(VI) Ln-xTB* re-fit)."""
    i = re.search(r"Table\s*S62", text).start()
    j = re.search(r"Re-optimized\s*parameters\s*of\s*Ce", text[i:]).start() + i
    seg = text[i:j]
    out = {}
    for zm in re.finditer(r"\$Z=(\d+)(.*?)\$end", seg, re.S):
        z = int(zm.group(1)); body = zm.group(2)
        d = {}
        for line in body.splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("ao=") or line.startswith("="):
                continue
            if re.match(r"^S\d+$", line):
                continue
            k, v = line.split("=", 1)
            vals = [_num(t) for t in v.split()]
            if k == "lev":
                d.update(lev1=vals[0], lev2=vals[1], lev3=vals[2])
            elif k == "exp":
                d.update(exp1=vals[0], exp2=vals[1], exp3=vals[2])
            else:
                d[k] = vals[0]
        assert set(d) == set(KEYS), (z, set(KEYS) - set(d))
        out[LN[z - 57]] = d
    assert len(out) == 15, len(out)
    return out


def fmt_block(z: int, d: dict[str, float]) -> str:
    L = [f"$Z={z} Ln-xTB (Zhang 2026, 10.1002/jcc.70321)",
         " ao=5d6s6p",
         f" lev= {d['lev1']:12.6f} {d['lev2']:12.6f} {d['lev3']:12.6f}",
         f" exp= {d['exp1']:12.6f} {d['exp2']:12.6f} {d['exp3']:12.6f}"]
    for k in KEYS[6:]:
        L.append(f" {k}={d[k]:12.6f}")
    L.append("$end")
    return "\n".join(L) + "\n"


def build(params: dict[str, dict[str, float]], stock_home: Path = STOCK_HOME) -> Path:
    src = stock_home / "share/xtb/param_gfn2-xtb.txt"
    text = src.read_text()
    for z in range(57, 72):
        pat = re.compile(rf"^\$Z={z}\b.*?^\$end\n", re.S | re.M)
        assert len(pat.findall(text)) == 1, z
        text = pat.sub(lambda _m: fmt_block(z, params[LN[z - 57]]), text)
    share = OUT / "share/xtb"
    share.mkdir(parents=True, exist_ok=True)
    (share / "param_gfn2-xtb.txt").write_text(text)
    for p in (stock_home / "share/xtb").glob("*"):
        if p.name != "param_gfn2-xtb.txt":
            shutil.copy(p, share / p.name)
    return share


def xtb_sp(xyz: Path, share: Path, charge: int, uhf: int = 0) -> float:
    env = {"XTBPATH": str(share), "OMP_NUM_THREADS": "1",
           "LD_LIBRARY_PATH": str(STOCK_HOME / "lib"), "PATH": "/usr/bin:/bin",
           "HOME": str(Path.home())}
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run([str(STOCK_HOME / "bin/xtb"), str(xyz), "--gfn", "2",
                            "--chrg", str(charge), "--uhf", str(uhf), "--sp",
                            "--norestart"], cwd=td, env=env,
                           capture_output=True, text=True, timeout=600)
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", p.stdout)
    if not m:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return float(m.group(1))


CHARGE = {"F3": 0, "Cl3": 0, "Br3": 0, "water": 3, "P507": 0, "C301": 0, "POO": 3}


def verify(share: Path, xyz_dir: Path, metals=("La", "Nd", "Gd", "Tb", "Lu")) -> list[dict]:
    rows = []
    for m in metals:
        for lig, q in CHARGE.items():
            f = xyz_dir / (f"{m}{lig}.xyz" if lig.endswith("3") else f"{m}_{lig}.xyz")
            if not f.exists():
                continue
            ref = float(re.search(r"energy:\s*(-?\d+\.\d+)", f.read_text().splitlines()[1]).group(1))
            e_ln = xtb_sp(f, share, q)
            e_g2 = xtb_sp(f, STOCK_HOME / "share/xtb", q)
            rows.append({"file": f.name, "charge": q, "si_energy": ref,
                         "lnxtb_sp": e_ln, "gfn2_sp": e_g2,
                         "d_lnxtb": e_ln - ref, "d_gfn2": e_g2 - ref})
            print(f"  {f.name:14s} q={q}  SI {ref:16.9f}  Ln-xTB {e_ln:16.9f} "
                  f"(d={e_ln - ref:+.2e})  stock GFN2 d={e_g2 - ref:+.3e}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--si-text", type=Path, help="pdftotext/pypdf dump of the SI")
    ap.add_argument("--verify", type=Path, help="dir with the SI's opted_xyz_by_Ln-xTB/*.xyz")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.si_text:
        text = args.si_text.read_text()
        tab, blk = parse_tables(text), parse_blocks(text)
        worst = max(abs(tab[m][k] - blk[m][k]) for m in LN for k in KEYS)
        print(f"[lnxtb] S60/S61 vs S62 cross-check: max |diff| = {worst:.2e}")
        assert worst < 1e-4, "the two SI copies disagree"
        (OUT / "lnxtb_params.json").write_text(json.dumps(
            {"source": "Zhang 2026 J Comput Chem 10.1002/jcc.70321, SI Tables S60-S62",
             "params": blk}, indent=1))
    params = json.loads((OUT / "lnxtb_params.json").read_text())["params"]
    share = build(params)
    print(f"[lnxtb] wrote {share / 'param_gfn2-xtb.txt'}")
    if args.verify:
        rows = verify(share, args.verify)
        (OUT / "verify.json").write_text(json.dumps(rows, indent=1))
        ok = max(abs(r["d_lnxtb"]) for r in rows)
        print(f"[lnxtb] max |E_sp - E_SI| = {ok:.2e} Eh over {len(rows)} structures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
