#!/bin/bash
#SBATCH --job-name=ln_pidiag
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pidiag_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pidiag_%j.err
#
# Cache the alpha-complex persistence diagrams once, then gate on them.
#
# A persistence image is a diagram plus a rendering of it.  Only the rendering
# depends on the hyperparameters being swept, so the 953 alpha complexes are
# built once here and every configuration afterwards is a matrix product
# (~1.9 s for all 953 at 128x128).  Without this the sweep would rebuild the
# alpha complexes ~50 times over.
#
# --strict checks every complex against src.geometry_features.persistence_diagram
# itself: filtering the cached points to PI_HOMOLOGY_DIMS must reproduce that
# function exactly.  The cache keeps the homology dimension, which the shipped
# function discards, and whether discarding it was right is one of the questions
# under test -- so the cache has to be provably the same object otherwise.
#
# --verify-against-shipped is the hard gate: rendering the cache at the shipped
# settings (resolution 20, spread 0.08, range (0, 2.5), H0+H1 summed, linear
# weight) must reproduce data/.../complex_gfn2xtb_pi_images.npz.  If it does not,
# the sweep must not run -- every number would be measuring something other than
# a retuned version of the published arm.
#
# CPU only, no GPU: this is gudhi and BLAS.  data/ is never written.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

echo "[pidiag] building diagram cache $(date -Is)"
python3 -m automl.qc.pi_sweep_build --build

echo "[pidiag] GATE: render at shipped settings, compare to shipped asset"
python3 -m automl.qc.pi_sweep_build --verify-against-shipped

echo "PIDIAG DONE $(date -Is)"
