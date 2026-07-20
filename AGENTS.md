# Codex instructions for lanthanide_dataset_builder

## Mission

Work as a careful research-software engineer on the lanthanide dataset and
geometry pipeline. Preserve scientific meaning, resumability, auditability, and
cluster safety. Prefer a small, focused change in the existing workflow over a
new parallel pipeline or a broad rewrite.

These instructions apply to the entire repository. A more specific `AGENTS.md`
in a subdirectory may add stricter rules for that subtree.

## First-pass orientation

Before changing anything:

1. Run `git status --short` and preserve all pre-existing user changes.
2. Read the files directly involved in the request and their tests. Do not scan
   generated geometry trees unless the task requires it.
3. Determine which stage owns the behavior:
   - stage 1 dataset/spec preparation: `scripts/build_dataset_no3d.py`;
   - chemistry planning: `src/chemistry/coordination.py` and `plan_complex()`;
   - stage 2 geometry generation, retry, recovery, and merge:
     `scripts/build_unique_geometries.py`;
   - cluster orchestration: `slurm/geometry_pipeline.slurm`;
   - schema and optional 3D joins: `src/geometry_schema.py`;
   - focused QC/regeneration helpers: existing scripts and reports, not a new
     standalone pipeline.
4. Use `rg` and `rg --files` for search. Because `data/` is often ignored, use
   `rg --files -u` only when the task explicitly requires generated artifacts.
5. Treat the current checkout as the source of truth. Historical commands,
   resource defaults, report names, and row counts may be stale.

## Non-negotiable safety boundaries

### Login node and Slurm

- Never run Architector geometry generation, conformer searches, quantum
  chemistry, large dataset builds, large parameter sweeps, or long/high-CPU
  Python jobs on a cluster login node.
- On a login node, only perform lightweight work: inspect files and logs, run
  syntax checks, prepare queues, use small metadata/report commands, inspect
  `squeue`/`scontrol`/`sacct`, and submit jobs with `sbatch`.
- Use `slurm/geometry_pipeline.slurm` as the single current stage-2 entrypoint.
  Do not revive old wrapper scripts without explicit evidence that they are
  required by the current checkout.
- Before a real submission, validate the exact route with:
  `DRY_RUN=1 bash slurm/geometry_pipeline.slurm <mode>`.
- A dry run is not permission to submit. Run real `sbatch` only when the user
  explicitly asks to launch/submit work.
- Never call worker modes manually unless debugging a specific documented
  failure and the command is demonstrably lightweight.
- Respect site QOS, account, partition, walltime, array, concurrency, CPU, and
  memory limits found in the current Slurm script or cluster output. Do not
  respond to a small rescue task with an unnecessarily large array.
- Remember that `sbatch` snapshots the script and resources at submission time.
  If behavior differs from local code, check the job submission time, job ID,
  cluster checkout, and branch before editing code again.
- Prefer resumable arrays, explicit dependencies, per-row/shard status, and a
  merge/report step that can describe partial completion. Do not hide incomplete
  work behind a successful exit status.

### Protected research data

- Treat `data/`, geometry files (`*.xyz`, `*.mol2`), generated indices, queues,
  and scientific result tables as read-only by default.
- Do not create, edit, delete, rename, reformat, regenerate, or overwrite
  anything under `data/` unless the user explicitly requests that data-producing
  operation. A user-approved Slurm pipeline may write its documented outputs.
- Never use generated data files as convenient fixtures for a code test. Use a
  temporary directory or a minimal synthetic fixture instead.
- Never delete or clean generated artifacts, even if ignored by Git, unless the
  user explicitly identifies the cleanup scope.
- Focused regeneration must preserve original geometries until review. Write to
  the existing isolated regeneration output path and keep accepted, rejected,
  and best-failed results distinguishable.
- `git pull` does not transfer ignored artifacts. When files are missing, first
  distinguish tracked CSV/report state from on-disk geometry artifacts; use an
  explicitly authorized transfer route such as `rsync` for the latter.

### Scientific integrity

- `plan_complex()` is the source of truth for coordination number, donor set,
  ligand count, fill ligand, and related stage-1 chemistry decisions.
- Preserve metal, oxidation state, `COORDLIST`, `DENTATE`, `coreCN`, `n_ligs`,
  ligand identity, and environment context through retry/rescue paths unless the
  user explicitly requests a scientifically justified change.
- Do not fabricate temperature-, time-, shaking-, solvent-, or condition-driven
  3D changes. Encode only relationships supported by the inputs or documented
  chemistry logic.
- Do not silently replace a failing ligand with simplified/fallback chemistry
  and report it as a normal success. Keep provenance and failure status explicit.
- Separate generation status from geometry QC. A produced XYZ is not
  automatically a chemically acceptable geometry.
- Do not hand-edit result CSVs to make counts or statuses look correct. Fix the
  producing/recovery logic and regenerate through an auditable path when the user
  authorizes it.

## Implementation rules

- Prefer the smallest coherent patch in an existing entrypoint. Avoid multiple
  new wrappers, duplicate queues, or a second geometry workflow.
- Preserve existing safety mechanisms in geometry generation: child-process
  isolation, timeouts, temporary write plus atomic replace, immediate index
  append/flush/fsync, resumability, and recovery/audit modes.
- Make skip, retry, overwrite, dependency, and partial-failure semantics explicit
  in code and user-facing output.
- For retry work, distinguish deterministic failures from retryable failures and
  rows with no XYZ from rows that produced an XYZ but failed QC.
- Keep the no-3D baseline intact when adding optional 3D features.
- Do not add a dependency or change environment setup unless necessary. Never
  expose credentials, tokens, private keys, or environment variable values.
- Do not make opportunistic unrelated refactors while fixing a focused issue.

## Validation

Use the least expensive validation that gives real evidence:

1. Inspect the final diff and ensure it contains only intended files.
2. For Python changes, run targeted tests from `tests/` and a syntax/import check
   that does not trigger heavy work or write research data.
3. For Slurm changes, run `bash -n slurm/geometry_pipeline.slurm`, then the
   relevant `DRY_RUN=1` route. Inspect printed resources, array bounds,
   dependencies, inputs, and output paths.
4. For recovery/merge changes, use temporary synthetic inputs where possible and
   verify incomplete work remains visible and returns the intended status.
5. Do not run the entire expensive pipeline merely to validate orchestration.
6. If local dependencies such as Architector, RDKit, Open Babel, or the cluster
   environment are unavailable, report that limitation precisely; do not claim
   validation that did not run.

## Reports and handoff

Every completed task must end with a concise operational report containing:

- outcome: what changed or what was diagnosed;
- scope: files changed, and confirmation that protected data was untouched;
- validation: exact checks run and their results;
- cluster state: dry-run only, submitted job IDs/dependencies, or explicitly
  "not submitted";
- remaining failures, risks, assumptions, and the shortest safe next command.

For long-running jobs, rely on persistent Slurm logs and the pipeline's existing
CSV/text report paths. Do not invent success from submission alone: submission,
running, completed, merged, and QC-accepted are separate states. If a new report
artifact is genuinely needed, put it under the established `reports/` structure,
make its provenance and inputs explicit, and do not overwrite an existing report
without user authorization.

## Git discipline

- Start and finish with `git status --short`; inspect `git diff` before handing
  off.
- Pre-existing modifications belong to the user. Never discard, overwrite,
  stage, or include them in a commit accidentally.
- Do not use destructive commands such as `git reset --hard`, `git clean`, or
  checkout-based file restoration unless the user explicitly requests the exact
  destructive operation.
- Do not stage, commit, amend, rebase, merge, push, open a pull request, or switch
  branches unless the user asks for that Git action.
- When asked to commit, stage only the intended files and use a descriptive
  commit message. When asked to branch, use the `codex/` prefix unless the user
  specifies another name.
- Before pushing, state the branch and remote. Never force-push without explicit
  user authorization.
- Do not assume a pushed feature branch changed the cluster checkout. Verify the
  branch/commit used on the cluster and whether an already queued job predates
  the change.
- Keep runtime logs, caches, environments, and generated artifacts out of commits
  unless the user explicitly asks to version a specific artifact. Do not alter
  tracked scientific data merely to obtain a clean working tree.

## Stop and ask before proceeding when

- the request would overwrite/delete research data or accepted geometries;
- a real cluster submission, cancellation, or large transfer was not explicit;
- chemistry identifiers or frozen coordination fields would need to change;
- the only available fix requires broadening scope into a pipeline rewrite;
- user changes overlap the same lines and cannot be preserved safely;
- credentials, a cluster-side state change, or external authorization is needed.

Otherwise, make a reasonable conservative assumption, implement the focused
change, validate it, and report the exact operational result.
