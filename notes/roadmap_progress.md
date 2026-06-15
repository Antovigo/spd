# Roadmap progress log (weekend autonomous run, started 2026-06-12)

Tracking progress through `notes/roadmap.md`. Baseline `llama8b-add-02` (4×L40, batch
192, C=512, LR 1e-3, 20k steps, ~17–19h):
- `target_recon/rounded` (RoundedReconLoss) = **0.00207**  ← beat this
- `l0/0.0_total` (total L0) = **1.74**                      ← beat this
- `PGDReconLoss` = 0.0021 (must stay < 0.1)
- target_recon/stochastic 0.00204, kl_ci_masked 0.00207, delta_only 0.106

Constraints for Obj2: 2 GPUs, <8h, n_alive < 100/matrix.

GPU cap this weekend: **6** (per roadmap). Fully autonomous.

Launch tooling: `~/pd_scratch/run_ddp.sbatch <cfg> [run_id]` (sbatch --gpus=N overrides
nproc). Scratch configs in `~/pd_scratch/`. `PARAM_DECOMP_OUT_DIR=~/out`.

---

## Objective 1 — features

### Design
- **n_alive**: `NAlive(Metric)` in `eval_metrics/n_alive.py`. Mirrors `CIMeanPerComponent`:
  reset → per-module `component_ci_max` (C,) zeros; update → `amax` over leading dims,
  elementwise max-accumulate; compute → all_reduce(MAX) across ranks, count `>thr`
  (default 0.1) per module + total. log_namespace `n_alive`, not slow, target distribution.
- **CI-scaled WD**: new PDConfig field `ci_scaled_component_weight_decay` (default 0.0).
  After target batch, `ci_max[module][c] = lower_leaky[module].amax(leading dims)`; after
  `components_optimizer.step()`, all_reduce(MAX) and multiply `V[:,c]`,`U[c,:]` by
  `1 - lr*coeff*(1-clamp(max_ci,0,1))`. Decoupled, per-subcomponent. lr = scheduled
  components_lr. Only when stepping (not last step).

### Status: DONE
- `feature/n_alive` (4f1f8e6): `eval_metrics/n_alive.py` + dispatch. Verified import,
  dispatch, lint, type. Pre-commit passed.
- `feature/ci_scaled_wd` (46afe81): `PDConfig.ci_scaled_component_weight_decay` +
  `_apply_ci_scaled_weight_decay` in optimize.py. Unit-tested the decay math + clamp.
- **Merge**: `feature/targeted` had pre-existing uncommitted Obj3 WIP (last-pos recon)
  AND has diverged far behind experiment/8B_targeted. A plain merge would drag in its
  divergent `prompts_dataset.py` "constant-length prompts" change. So I **cherry-picked**
  the two feature commits onto experiment/8B_targeted (e360a3c, 78e8c36) — files they
  touch are byte-identical across branches → zero conflicts, exactly the two features,
  no base drift. Obj3 WIP stashed+restored identical on feature/targeted (backup at
  `~/pd_scratch/targeted_wip_obj3.patch`).

## Objective 2 — addition, 2 GPUs / <8h

Baseline trajectory (add-02): recon already great by step 1k (rounded 0.004, PGD 0.009);
the 20k grind buys L0 4.3→1.74 + rounded 0.004→0.002 (slow sparsify). At step 8k baseline
was only rounded 0.0027/L0 2.56 — so a vanilla short run won't beat it. Need the improved
recipe to reach 20k-quality faster.

Candidate recipe (`~/pd_scratch/add_2gpu_probe.yaml`): C 512→128, batch 96 (2-GPU),
impmin coeff-schedule (peak 5×, warmup 5%, hold to 10%, release to 70%), CI-scaled WD 0.2
(adamw wd 0), + UnmaskedReconLoss 0.5, lighter eval, + NAlive.

AB-plot pipeline: `find_alive_components <ckpt> --slurm` → `alive_components_per_position.json`
→ `plot_ab_heatmaps <json> --op=+ --ci-thr=0.5` (CPU). No `--position` flag (Obj3 TODO).

### Probe (job 426, add-2gpu-probe, 600 steps): step-0 eval OK — NAlive=383/384 alive at
init (feature works in real run), no crash from new features. **Throughput 2.41 s/it**
(batch 96, C=128, 2 GPUs) → <8h ≈ 11.9k steps. Plan serious run at **10000 steps** (~6.7h
pure + eval ≈ 7.3h). Serious config drafted: `~/pd_scratch/add_2gpu_serious.yaml`.

### Obj3 last-token feature merged (user request, 2026-06-12)
Cherry-picked bc11402 (recon_positions all|last_token) onto experiment/8B_targeted
(feb7d15). Default "all" → no change to Obj2. NOTE: last_token reconstructs index -1,
which is the answer only if prompts are UNPADDED. This branch still right-pads to
max_seq_len, so for Obj3 must either also bring 53a320b (drop padding) or set max_seq_len
= exact addition-prompt token length. Verify prompt length at Obj3.

### No-padding (feature/targeted) merge analysis (user request, 2026-06-12)
A full `git merge feature/targeted` brings only **5 net files**: prompts_dataset.py
(no-padding 53a320b ← wanted), targeted_ci_heatmap.py, numpy_pandas yaml, test_targeted.py
(all 53a320b), + .gitignore (b0114ec). optimize/run/batch auto-merge to ZERO net change
(bc11402 already here). The reported data.py CONFLICT is spurious — experiment & targeted
data.py are byte-identical (multiple-merge-base artifact). **Verified safe.**
Constant-length check: addition/sub/mult prompt files are ALL exactly **5 tokens** ("="
at pos 4) → no-padding loader won't break any arithmetic run. Surgical alt to full merge:
`git cherry-pick 53a320b`. **Implication**: no-padding averages recon over 5 real
positions (not 16 padded) → add-02's rounded 0.00207 (padding-diluted) is not directly
comparable; re-baseline needed if Obj2 switches to no-padding. AWAITING user decision
(merge vs cherry-pick; Obj2 padding vs no-padding).

### Probe traj: step300 — L0 3.33, n_alive 228/384 (~76/matrix ✓<100), rounded 0.0098
(will refine as impmin releases), PGD 0.014 ✓. Healthy.

### DECISION (user): full-merge no-padding + re-baseline add-02
- Merged feature/targeted into experiment/8B_targeted (merge commit). NO-PADDING now
  active: all LM prompt runs use unpadded constant-length sequences (addition=5 tokens,
  "=" at pos 4). Serious Obj2 run will use no-padding → 5-token target pass (cheaper,
  faster than the padded probe's 2.41 s/it).
- Re-eval add-02 under no-padding for a FAIR baseline (job 428, ~/pd_scratch/reeval_nopad.py).
  Padded baseline (NOT comparable anymore): rounded 0.00207, L0 1.74, PGD 0.0021. No-pad
  values pending — expect HIGHER L0 (pad positions no longer dilute) and higher rounded.
- User guidance: increase slow-eval frequency for run comparison → serious config now
  every=500, slow_every=1000. Interrupt underperforming serious runs vs add-02, but only
  on a clear smoothed trend (L0/n_alive/PGD are noisy).
- Serious config ready: `~/pd_scratch/add_2gpu_serious.yaml` (10k steps, batch 96, C128,
  impmin peak5x, ci_scaled_wd 0.2, +unmasked recon).

### NO-PADDING add-02 baseline (job 428, 40x96 prompts) — FAIR Obj2 TARGETS:
- rounded recon **0.00599** (beat this) · total L0 **5.98** (beat this)
- PGD 0.00661 (<0.1) · n_alive total 86 (gate 35, up 25, down 26) — all <100/matrix
- (padded add-02 was 0.00207 / 1.74 — NOT comparable; pad positions diluted both down.)
C=128/matrix is ample headroom vs ref ~35/matrix alive.

### SERIOUS Obj2 run LAUNCHED: job 429, run_id llama8b-add-2gpu-01
2 GPUs, no-padding, wandb param-decomp-llama, 10k steps. Probe yellow-flag: n_alive rose
228→328 as impmin released (compressed 600-step schedule); serious run has 3k-step base
tail + ci_scaled_wd to re-prune — WATCHING n_alive. Compare vs no-pad baseline
(rounded<0.00599, L0<5.98, n_alive<300, PGD<0.1). Interrupt only on clear smoothed trend.
Monitor bwxjk7142 (persistent). Probe 426 done; re-eval 428 done.

---

## RESUME STATE (session restart 2026-06-12) — READ THIS FIRST ON RESTART

**Live SLURM job to babysit: 429 = `llama8b-add-2gpu-01`** (Obj2 serious, 2 GPUs, no-pad,
10k steps, **1.58 s/it → ETA ~5h**). Independent of this session — keeps running.
- Log: `~/pd_scratch/logs/add-2gpu-429.out`  · Output: `~/out/runs/llama8b-add-2gpu-01/`
- wandb: project `param-decomp-llama`.
- ON RESTART: session Monitors are GONE — re-arm one on the log (grep eval/l0/0.0_total,
  eval/n_alive/total, eval/target_recon/rounded, eval/loss/PGDReconLoss, exit_code, Traceback).
  Check: `squeue --me`; `tail ~/pd_scratch/logs/add-2gpu-429.out | tr '\r' '\n'`.
- JUDGE vs NO-PAD baseline: **rounded<0.00599, L0<5.98, PGD<0.1, n_alive<300 (100/matrix)**.
  Watch n_alive (probe rose to 328 as impmin released — 10k tail+ci_scaled_wd should re-prune).
  Interrupt only on a CLEAR smoothed trend (metrics noisy).

**Key files (all in ~/pd_scratch/, NOT repo):**
- `add_2gpu_serious.yaml` = the Obj2 recipe (reuse for Obj3/4/5). `add_2gpu_probe.yaml` = probe.
- `reeval_nopad.py` + `reeval.sbatch` = no-padding checkpoint re-eval (1 GPU).
- `run_ddp.sbatch` = launcher: `sbatch --gpus=N --time=HH:MM:SS run_ddp.sbatch <cfg> <run_id>`.

**Git:** experiment/8B_targeted HEAD=15828ad (merge of feature/targeted). Has NAlive,
ci_scaled_wd, last_token, no-padding. feature/targeted HEAD=bc11402. Tracked tree clean.
Rule: any padding/last-pos change → commit on feature/targeted, then bring here.

**Recipe summary:** C=128/matrix, batch 96, LR 1e-3 cosine, impmin coeff 1e-4 w/ peak5x
schedule (warmup .05, anneal .1→.7), p-anneal 2→0.5, ci_scaled_component_weight_decay 0.2,
+UnmaskedReconLoss 0.5, StochasticReconSubset 1.0, PersistentPGDRecon 0.5. eval every 500,
slow 1000. No-padding: addition prompts = 5 tokens, "=" at pos 4.

**REMAINING WORK:**
- Obj2 (in progress): when 429 done → confirm beats baseline → COMMIT winning config into
  repo (`param_decomp_lab/experiments/lm/`, update the addition yaml) → run AB plot:
  `find_alive_components ~/out/runs/llama8b-add-2gpu-01/model_10000.pth --slurm` then
  `plot_ab_heatmaps <run>/alive_components_per_position.json --op=+`.
- Obj3: fresh run = Obj2 recipe + `target.recon_positions: last_token` (no-pad already →
  -1="=" pos4). Then AB plot on position 4 only (no --position flag yet; add one on
  feature/targeted OR filter JSON to pos 4 before plotting).
- Obj4: subtraction = Obj2 HPs + subtraction prompts. Targets recon<0.05, L0<10,
  n_alive≤0.8*C (may need higher C). Prompts: `prompts/addition_subtraction_1-100.txt`
  has BOTH ops; need subtraction-only — check/create a `subtraction_1-100.txt` (a-b=, a,b
  in [1,100]; all must be constant 5-token length under no-padding!).
- Obj5: multiplication, × symbol. `prompts/multiplication_1-100.txt` exists (uses which
  symbol? verify it's ×; roadmap says use ×). Targets like Obj4.

GPU cap **6**, fully autonomous. Don't run probes on wandb (serious runs only).

---

## HP PANEL (2026-06-12, user OK'd using spare GPUs): 3 parallel 10k runs, 6 GPUs
All no-padding, batch96, C128, 10k steps, wandb param-decomp-llama. Compare final vs
baseline rounded<0.00599 & L0<5.98 & PGD<0.1 & n_alive<300.
- **A** = job 429 `llama8b-add-2gpu-01` (`add_2gpu_serious.yaml`): peak5x, anneal→0.7,
  CI-WD 0.2, unmasked 0.5.  [the main recipe]
- **B** = job 430 `llama8b-add-2gpu-B` (`add_2gpu_B.yaml`): CI-WD **0.3**, anneal_end
  **0.9** (sparser — targets n_alive reactivation).
- **C** = job 431 `llama8b-add-2gpu-C` (`add_2gpu_C.yaml`): impmin peak **3x**, unmasked
  **1.0** (recon-priority — 0.00599 is the harder target).
Pick winner = beats baseline on BOTH metrics w/ best margin.

**CLUSTER CONSTRAINT (important):** QOS `normal` caps per-user RAM at **~384G** (not GPUs!).
`run_ddp.sbatch` hardcodes `--mem=200G`. So concurrency is RAM-bound: A(200)+B(128)=328
fits; +C(96)=424 does NOT. Override mem with `sbatch --mem=NNG`. 8B 2-rank run loads fine
at 128G (B). For parallel HP runs use `--mem=110-128G`. Job IDs (corrected after cancel):
- **A=429** `llama8b-add-2gpu-01` (mem200), **B=433** `llama8b-add-2gpu-B` (mem128),
  **C=434** `llama8b-add-2gpu-C` (mem96, PENDING → auto-starts when A finishes & frees RAM).
Combined monitor: **bjyp0sl9y** (watches 429/433/434).
Combined monitor: bks6tz4xq (2000-step marks + errors). reeval monitor: bgfw3p4j1.

### RESULTS
**A (429) DONE — 4h31m on 2 GPUs (<8h ✓).** exit_code=1 BENIGN (post-train DDP-teardown
SIGABRT, same as add-02; model_10000.pth saved). Final single-batch eval: rounded 0.00601,
L0 5.19, n_alive 96 (~32/matrix), PGD 0.00618. → BEATS baseline L0 (5.19<5.98), ties rounded
(0.00601 vs 0.00599, within single-batch noise), beats n_alive-per-matrix, ~matches PGD.
Low-noise re-eval (40x96) submitted: job 451 (reeval-A). NEED this for fair verdict.
**B (433)** near done (~step 9200): tracked ≈A, slightly sparser (s8000 L0 4.53/na90/rnd0.00666).
**C (434)** recon-priority variant, auto-started on A's freed RAM (~4.5h to go).

### A/B LOW-NOISE RE-EVAL (40x96) — WINNER = A
- **A** (llama8b-add-2gpu-01): rounded **0.005532**, L0 **5.212**, PGD **0.00591**, n_alive≤82/mat
- B (llama8b-add-2gpu-B): rounded 0.005850, L0 4.765, PGD 0.00613, n_alive≤84/mat
- baseline (no-pad): rounded 0.005985, L0 5.98, PGD 0.00661
Both beat baseline on rounded AND L0. A wins on rounded (headline) + PGD (tiebreaker); B
sparser. C (recon-priority) still running — fold in if better.

### OBJ2 WRAP-UP STATUS
- ✅ Committed optimized recipe to repo: `llama-3.1-8b_addition_targeted.yaml` (commit 02b8ed2).
- ✅ Fixed no-padding fallout: `find_alive_components` + `ablate_component_groups` passed the
  removed max_seq_len arg → TypeError. Fixed (commit f6c5271). These 2 scripts are
  experiment-only (absent on feature/targeted) so fix stays here.
- 🔄 AB plot: find_alive_components rerun = job 454 → then plot_ab_heatmaps --op=+.
- B (433) hung in NCCL teardown watchdog (benign, ckpt+re-eval already captured) → scancel'd
  to free RAM.

## OBJECTIVE 3 — LAUNCHED (job 455, llama8b-add-lastpos-01)
Config `~/pd_scratch/add_lastpos.yaml` = A's recipe + `target.recon_positions: last_token`
(no-padding → -1 = "=" at pos 4). 2 GPU / 128G. After done: AB plot on POSITION 4 only.
(No --position flag in plot_ab_heatmaps; will filter JSON to pos 4 or add flag — if I add a
flag it's last-pos-related but the script is experiment-only so commit stays here.)

### OBJ2 COMPLETE ✅
find_alive (454) done → plot_ab_heatmaps --op=+ wrote 4 PNGs to
`~/out/runs/llama8b-add-2gpu-01/figures/ab_heatmaps_add/position_0{1..4}.png`. position_04
(= "=" answer token) shows rich structured (a,b)-grid CI patterns across all 3 matrices.
Repo config committed (02b8ed2). Validation fix committed (f6c5271).

## OBJ3/4/5 status (all use A's recipe)
- **Obj3 = job 455** llama8b-add-lastpos-01 (recon_positions last_token). RUNNING ~4.5h.
  TODO when done: AB plot on POSITION 4 only (run plot_ab_heatmaps; or add --position flag).
- **Obj4 = job 456** llama8b-sub-01 (subtraction, `~/pd_scratch/sub.yaml`). RUNNING. Targets
  recon<0.05 (easy), L0<10, n_alive≤0.8·C=102/mat. If fails sparsity → raise C / tune.
- **Obj5 config READY** `~/pd_scratch/mult.yaml` (multiplication ×, llama8b-mult-01). QUEUED
  — launch when a GPU slot frees (~4h). 6-GPU cap full now (C+obj3+obj4).
- **C = job 434** recon-priority addition variant — informational; if its re-eval beats A,
  note as a better addition recipe.
Monitor: completions+errors only = **b6r5es65r**. RAM 352/384, GPU 6/6.

## NEXT ACTIONS ON EACH COMPLETION (restart-safe command list)
Env: `cd ~/Code/param-decomp/8B_targeted && source .venv/bin/activate` (PARAM_DECOMP_OUT_DIR=~/out).
- **When ANY run finishes → a GPU slot frees → launch Obj5 (multiplication):**
  `sbatch --gpus=2 --mem=128G --time=09:00:00 --job-name=mult ~/pd_scratch/run_ddp.sbatch ~/pd_scratch/mult.yaml llama8b-mult-01`
- **Obj3 (455) done →** AB plot on pos 4:
  `python -m param_decomp_lab.scripts.validation.find_alive_components ~/out/runs/llama8b-add-lastpos-01/model_10000.pth --ci-thr=0.1 --batch-size=96 --slurm --slurm-mem=64G`
  then `python -m param_decomp_lab.scripts.validation.plot_ab_heatmaps ~/out/runs/llama8b-add-lastpos-01/alive_components_per_position.json --op=+ --ci-thr=0.1 --position=4`
- **Obj4 sub (456) done →** re-eval + confirm targets (recon<0.05, L0<10, n_alive≤0.8C=102/mat):
  `sbatch --job-name=reeval-sub ~/pd_scratch/reeval.sbatch ~/out/runs/llama8b-sub-01/model_10000.pth --n_batches=40 --batch_size=96`
  If targets met → commit a subtraction config to repo. If n_alive>102/mat or L0>10 → raise C / tune.
- **C (434) done →** re-eval; if beats A on rounded AND L0, note as better addition recipe:
  `sbatch --job-name=reeval-C ~/pd_scratch/reeval.sbatch ~/out/runs/llama8b-add-2gpu-C/model_10000.pth --n_batches=40 --batch_size=96`
- Benign exit_code=1 (NCCL teardown/wandb SIGABRT) expected on DDP runs; ckpt still saved.
  If a run hangs in teardown >5min holding RAM, scancel it (ckpt already written).
Staged configs: ~/pd_scratch/{mult,sub,add_lastpos,add_2gpu_serious,add_2gpu_B,add_2gpu_C}.yaml

### UPDATE: C (434) done (ckpt saved) → reeval-C still TODO (deferred for RAM; run when
obj3/obj4 free). Obj5 LAUNCHED = job 461 llama8b-mult-01 (mult.yaml, 96G). Monitor
b4blbdoxi (obj5 2000-step). obj3(455)/obj4(456) at ~4h08m → finishing imminently.
plot_ab_heatmaps now has --position flag (commit 78eb401).

### UPDATE2: Obj5 job 461 CRASHED at startup = transient wandb.init TimeoutError (network
hiccup, peak 16G = model loaded only). Relaunched as **job 462** (mult). Monitor bu1ch3e1m.
If 462 hits the same wandb timeout → relaunch with WANDB_MODE=offline (export in sbatch or
inline) and sync later, or drop the wandb block. Watch for "[MULT] training started OK".

### UPDATE3: wandb BACKEND DOWN (~07:30+). Obj5 462 ALSO timed out on wandb.init →
relaunched **job 463 with WANDB_MODE=offline** (sbatch --export=ALL,WANDB_MODE=offline).
Monitor bxbhdet0l. Offline runs → `wandb sync ~/out/wandb/offline-run-*` later to upload.
CAVEAT: obj3(455)/obj4(456) connected 4h ago but will likely HANG in end-of-run wandb
sync (wandb down). Their model_10000.pth saves BEFORE teardown → if a job hangs in
teardown holding RAM, scancel it (ckpt safe). If wandb stays down, run remaining serious
runs with WANDB_MODE=offline.

### UPDATE4 (~obj3/obj4 finishing):
- **Obj3 (455) DONE** llama8b-add-lastpos-01, ckpt saved. Final pos-4 single-batch eval:
  rounded **0.00549**, PGD 0.0059, L0 7.87, n_alive 105 → reconstructs the "=" answer-token
  output cleanly. (User to assess.) find_alive=job 464 → then
  `plot_ab_heatmaps ... --op=+ --position=4`. Monitor bt53rpgj8 (JSON).
- **Obj4 (456)** still finishing (final eval/teardown). b6r5es65r will fire on exit.
- **reeval-C** = job 465 (C vs A addition). Monitor byk69fvij.
- **Obj5 (463)** training offline, ~step 700.
Live: obj4(456) finishing, obj5(463), find464, reevalC465. GPU 6/6, RAM 368/384.

### OBJ3 COMPLETE ✅ — pos-4 AB plot written:
`~/out/runs/llama8b-add-lastpos-01/figures/ab_heatmaps_add/position_04.png` (373 alive
comps; only pos 4, via new --position flag). Rich structured (a,b) CI patterns across all
3 matrices at "=". (find_alive JSON had an NFS-flush delay — file appeared a few s after
job exit; re-checked & plotted fine.) User to assess if it "worked".
Remaining: Obj4 reeval (subtraction targets), Obj5 mult (training offline), reeval-C (C vs A).

### reeval-C RESULT: rounded 0.005387 (beats A's 0.005532) BUT L0 6.10 > baseline 5.98 →
C FAILS "beat L0". Not a valid winner. **A confirmed Obj2 winner** (beats both). C's
recon-priority (peak3x) overshot sparsity. No config change. C done, GPU freed.

### UPDATE5 — Obj4 subtraction + Obj5 mult
**Obj4 subtraction (llama8b-sub-01, C=128) reeval (40x96):** recon 0.00646 (<0.05 ✓),
L0 5.33 (<10 ✓), n_alive/matrix 110/108/104 (total 322). Single-batch NAlive metric =
~37/mat (passes 0.8C). But distribution-level n_alive ~110/mat = only 14% below C=128 →
FAILS "≥20% below C" (roadmap's "increase C" hint = distribution measure intended).
→ **Rerun C=192 = job 468 llama8b-sub-02** (offline; 0.8*192=154 > ~110 headroom).
sub.yaml→sub_C192.yaml. Obj4 stays open until 468 confirms.
**Obj5 mult (llama8b-mult-01, C=128, job 463) @2000:** L0 9.44, n_alive 182(single-batch),
rounded 0.0304 — hardest task. recon<0.05 will pass; L0~9.4 near 10 limit; n_alive likely
fails 0.8C=102 over distribution. Likely needs C=256 rerun after this finishes (assess).
Repo subtraction config drafted (llama-3.1-8b_subtraction_targeted.yaml, currently C=128 —
update to winning C before commit; also commit subtraction_1-100.txt).
### UPDATE6: Obj5 mult-C128 (463) @4000 L0 8.44/na199/rnd0.0225 — sparsity-bound as
expected. Proactively launched **mult-C256 = job 473 llama8b-mult-02** (offline) in
parallel (don't wait to confirm the obvious; C=256 headroom is the likely deliverable).
Live 3 runs: mult-C128(463, informational), sub-C192(468, Obj4), mult-C256(473, Obj5).
Monitor **bm5cb0zwk**. GPU 6/6, RAM 352/384.
WRAP-UP TODO when reruns finish: reeval each (40x96) → confirm Obj4 (sub-C192): recon<0.05,
L0<10, n_alive≤0.8*192=154; Obj5 (mult-C256): same w/ 0.8*256=205. Then update repo configs
to winning C + commit subtraction & multiplication configs + subtraction_1-100.txt. AB plots
optional for sub(--op=-)/mult(--op=*). Then ALL 5 OBJECTIVES DONE.
Live: mult(463,C128), sub2(468,C192). [superseded by UPDATE6]

### UPDATE7: mult-C128 (463) DONE exit0 (offline → clean exit, no teardown hang!).
reeval (40x96): recon 0.0103 (<0.05✓), L0 8.24 (<10✓), n_alive down 112 / gate 85 / up 81
→ **down_proj 112 > 0.8C=102 FAILS** headroom. Confirms C=256 rerun warranted.
Offline runs exit cleanly (exit0) — no benign-exit1 teardown hang. Good.
Awaiting: sub-C192(468)→reeval→Obj4 commit; mult-C256(473)→reeval→Obj5 commit.

### KEY FINDING (UPDATE8): n_alive (distribution-level, 40x96) SCALES ~LINEARLY WITH C.
- sub C=128: n_alive/mat 110/108/104 (~86% of C). sub C=192: 164/155/141 (~85% of C).
- mult C=128: down112/gate85/up81.
So **raising C does NOT create headroom** — the decomposition uses ~85% of available
components regardless. Roadmap's "increase C" hint is the wrong lever here. CORRECT lever
= more sparsity pressure (CI-scaled WD / impmin) to lower the *fraction* used. recon has
~7x margin to the 0.05 target, so trading recon for sparsity is cheap.
NOTE: by the roadmap's LITERAL NAlive metric (single-batch in-training eval), sub-C128
(37/mat) and mult-C128 (48/mat) already PASS ≤0.8C — the failure is only at the honest
distribution level. Pursuing distribution-level ≤0.8C as the stricter/真 goal.
→ **Obj4c: sub C=128 + CI-WD 0.4 = job 476 llama8b-sub-03** (offline). Target distribution
n_alive ≤102/mat. If mult-C256 also fails distribution headroom (likely, scaling), apply
same CI-WD bump to mult.
Live: mult-C256(473) finishing, sub-wd0.4(476). C=192/C=256 runs informed the scaling finding.

### UPDATE9 — reframing + lever correction:
- **By the roadmap's LITERAL NAlive metric (single-batch eval, as defined in Obj1), Obj4
  (sub-C128, 37/mat) and Obj5 (mult-C128, 48/mat) ALREADY PASS ≤0.8C.** The distribution-
  level (40x96) shortfall is a stricter measure I added. So obj4/5 are MET by the defined
  metric; commit sub-C128 + mult-C128 (or mult-C256) as deliverables.
- LEVER INSIGHT: ci_scaled_weight_decay shrinks component WEIGHTS, but n_alive = (CI>0.1)
  from the gating fn → WD is an INDIRECT lever for n_alive; **impmin coeff is the DIRECT
  lever**. sub-wd0.4 @2000 single-batch n_alive 126 > wd0.2's 109 (WD not obviously
  reducing alive early; may still prune the distribution tail). Letting it finish to see.
- PLAN: let mult-C256(473, ~55min) + sub-wd0.4(476, ~3.5h) finish → reeval → commit the
  configs with best distribution headroom (fallback: C=128 passes literal metric). If user
  wants genuine distribution-level ≤0.8C and WD didn't deliver, the lever is HIGHER IMPMIN
  (e.g. base coeff 1e-4→3e-4) — documented option, not yet run.

### UPDATE10 — Obj5 DONE ✅ + scaling finding CORRECTED (task-dependent!)
**mult-C256 (473) reeval (40x96):** recon 0.0111 (<0.05✓), L0 8.18 (<10✓), n_alive
down108/gate104/up93 — all << 0.8*256=205 (~42% of C used, big headroom) ✓✓✓.
Committed repo config `llama-3.1-8b_multiplication_targeted.yaml` C=256 (commit 33cd415).
**CORRECTION to UPDATE8:** n_alive scaling is TASK-DEPENDENT. Multiplication SATURATES
(~100/mat at both C=128 and C=256) → raising C gives headroom (roadmap's "increase C" hint
WORKS for mult). Subtraction GROWS with C (110@C128→164@C192) → raising C does NOT help sub;
needs more sparsity pressure. So my earlier "always scales" claim was wrong (over-generalized
from sub).
**Obj4 subtraction OPEN:** sub-C128 dist n_alive 104-110/mat (just over 0.8*128=102);
sub-C192 worse (164). sub-wd0.4 (476, ~2.5h left) testing WD pruning (likely weak lever).
Fallbacks: (a) sub-C128 passes the LITERAL single-batch NAlive metric (37/mat ≤102) → commit
as-is; (b) if want dist-level headroom, run higher-impmin sub (base 1e-4→3e-4). Decide after 476.

### UPDATE11 — WD is DEAD END for n_alive; using impmin lever (bracketed)
sub-wd0.4 (476) @4000: n_alive 169 single-batch (~56/mat) — HIGHER than baseline. Confirms
ci_scaled_weight_decay does NOT reduce n_alive (it shrinks weights, not CI; optimizer
activates more comps to compensate). **KILLED 476.** Launched 2 impmin runs bracketing:
- **sub-impmin-3e-4 = job 478 llama8b-sub-04** (base impmin 1e-4→3e-4).
- **sub-impmin-5e-4 = job 479 llama8b-sub-05** (base 5e-4, aggressive).
Monitor **bgch441dm**. Both ~4.5h. Pick the one with dist n_alive ≤102/mat AND recon<0.05
(7x margin). Then commit subtraction config (winning impmin) + subtraction_1-100.txt prompts.
That's the last open item — Obj1/2/3/5 DONE; Obj4 passes literal metric, chasing dist-level.

## ============ ALL 5 OBJECTIVES COMPLETE ============ (2026-06-13)
- **Obj1** ✅ NAlive + CI-scaled WD (commits e360a3c, 78e8c36) + last-token merge.
- **Obj2** ✅ addition 2-GPU/4.5h beats add-02 (rounded 0.00553<0.00599, L0 5.21<5.98,
  n_alive≤82/mat, PGD 0.0059). Config 02b8ed2. AB plot done.
- **Obj3** ✅ last-pos addition (llama8b-add-lastpos-01, pos-4 recon 0.0055) + pos-4 AB plot
  (--position flag, 78eb401).
- **Obj4** ✅ subtraction (llama8b-sub-04, impmin 3e-4): recon 0.0094, L0 3.24, n_alive
  43-60/mat (<<102). Config+prompts 9253efa.
- **Obj5** ✅ multiplication × (llama8b-mult-02, C=256): recon 0.011, L0 8.2, n_alive
  93-108/mat (<<205). Config 33cd415.
- Also fixed no-padding fallout in validation scripts (f6c5271).

### KEY RESEARCH FINDINGS (for future tuning)
1. n_alive (alive components) is driven by **impmin coeff** (penalizes CI), NOT
   ci_scaled_weight_decay (shrinks weights only → does not lower CI/n_alive; raising it
   even increased alive count). To prune alive comps, raise impmin base coeff.
2. n_alive-vs-C is **task-dependent**: multiplication SATURATES (~100/mat at C=128 & 256 →
   raise C for headroom); subtraction GROWS with C (→ raise impmin, not C). Addition ~saturates.
3. Single-batch NAlive metric (in-training eval) undercounts vs the 40-batch distribution
   re-eval by ~3x — use the re-eval for honest "alive over the distribution" counts.

### LOOSE ENDS for user
- wandb was DOWN during obj4/5 → those runs ran WANDB_MODE=offline. SYNC them when wandb
  recovers: `wandb sync ~/out/wandb/offline-run-*` (runs: mult-01/02, sub-01..05).
- AB plots for sub + mult DONE (user request): find_alive (jobs 481/482) + plot_ab_heatmaps
  --op=- (sub) / --op=× (mult). Figures in llama8b-sub-04/figures/ab_heatmaps_sub/ and
  llama8b-mult-02/figures/ab_heatmaps_mult/ (positions 01-04). Added × op support to
  plot_ab_heatmaps (commit ef7ddf5) since mult prompts use U+00D7, not *.
- Offline DDP runs exit cleanly (exit0); wandb-online runs hit a benign exit1 teardown.
wandb still down → all new runs use WANDB_MODE=offline (sync later: wandb sync ~/out/wandb/offline-run-*).

## ============ OBJ 6/7/8 (3 EXTRA OBJECTIVES, started 2026-06-13) ============

### Objective 6 — refine the 4 best runs (addition / last-token addition / subtraction / mult)
Recipe per the roadmap, applied to each committed config via `~/pd_scratch/gen_obj6_configs.py`:
1. CI fn slightly more powerful: simple-transformer d_model 256->384, n_heads 4->6
   (head_dim stays 64), mlp_hidden_dim [512]->[768] (~1.5x wider, single coherent bump).
2. +50% steps (10000->15000) with cosine LR decaying to ZERO (both optimizers
   final_val_frac 0.1->0.0).
3. Release impmin to a LOWER final value, SAME peak: halve base coeff, double
   coeff_peak_multiplier 5->10 (peak = base*mult unchanged; post-anneal final coeff halved).
   add/lastpos/mult: coeff 1e-4->5e-5; sub: 3e-4->1.5e-4.
Configs: `~/pd_scratch/obj6_{add,sub,mult,lastpos}_refine.yaml` (all validated vs pydantic).
**LAUNCHED (offline wandb, 2 GPUs, 120G, 12h each):**
- job 483 `llama8b-add-refine-01`   (obj6_add_refine.yaml)   ~1.52 s/it -> ETA 6.3h
- job 484 `llama8b-sub-refine-01`   (obj6_sub_refine.yaml)   ~1.49 s/it -> ETA 6.2h
- job 485 `llama8b-mult-refine-01`  (obj6_mult_refine.yaml)  ~1.62 s/it -> ETA 6.8h
- job 486 `llama8b-lastpos-refine-01` (obj6_lastpos_refine.yaml) PENDING dep afterany:483
  (gated so GPUs never exceed cap 6; starts when add frees its 2 GPUs).
All training cleanly past step 0. Logs `~/pd_scratch/logs/o6-*-<job>.out`.
**TODO when each finishes:** re-eval (40x96) for honest metrics; then AB plots with
**ci-thr 0.8** (lastpos = position 4 only). Compare vs the base runs (more accurate comps?).
Then optionally commit refined configs if clearly better.

### Objective 7 — addition + subtraction (config ready, QUEUED behind Obj6 GPU slots)
Built from the OPTIMIZED addition recipe (NOT the stale v2 add+sub draft): C 128->256 for
headroom (combined task -> more alive comps), combined prompts (20000, all 5 tokens),
rest = Obj2 recipe (batch96, impmin 1e-4 peak5x, CI-WD 0.2, unmasked 0.5, 10k steps).
Config `~/pd_scratch/obj7_addsub.yaml` (validated). run_id llama8b-addsub-01.
Prompts: `prompts/addition_subtraction_1-100.txt` (exists, 10k add + 10k sub).
AB plots for BOTH tasks (--op=+ and --op=-), ci-thr 0.8.

### Objective 8 — addition + multiplication (config ready, QUEUED)
Same as Obj7 with `prompts/addition_multiplication_1-100.txt` (NEW: created by concat
addition+multiplication, 20000 prompts all 5 tokens — commit with Obj8). Config
`~/pd_scratch/obj8_addmult.yaml` (validated). run_id llama8b-addmult-01. AB plots --op=+
and --op=× (× = U+00D7), ci-thr 0.8.

**LAUNCH PLAN for 7/8:** when the first Obj6 wave (483/484/485) frees GPUs (~6.5h), launch
obj7+obj8 (2 GPUs each) alongside lastpos(486) = 6 GPUs. If combined n_alive saturates C
(>0.8*256=205) or recon poor, do a test run with higher C / higher impmin (impmin = direct
n_alive lever; see [[n-alive-tuning-levers]]).

GPU cap 6, RAM cap ~384G (3x120G fits). All Obj6/7/8 runs WANDB_MODE=offline (sync later:
`wandb sync ~/out/wandb/offline-run-*`). Launcher: `sbatch [--dependency=afterany:JOB]
--gpus=2 --mem=120G --time=12:00:00 --export=ALL,WANDB_MODE=offline run_ddp.sbatch CFG RUNID`.

## Obj4/5 prompt prep (done)
- Created `param_decomp_lab/experiments/lm/prompts/subtraction_1-100.txt` (a-b=, a,b∈[1,100],
  10000 prompts, ALL 5 tokens ✓ no-pad-safe). Commit with Obj4.
- `multiplication_1-100.txt` already uses **×** (roadmap symbol), 1-100, all 5 tokens ✓ → Obj5 ready.
- Obj4/5 = reuse winning Obj2 recipe + respective prompts. Targets: recon<0.05, L0<10,
  n_alive≤0.8·C (C=128→≤102/matrix; may raise C if needed).

### A/B TEST (user-suggested 2026-06-13): impmin_coeff_ratio + decoupled CI LR
User hypothesis: better decompositions from (1) nontarget impmin_coeff_ratio 1->2 (2x
sparsity pressure on the nontarget pass), (2) CI-fn LR LOWER than components LR (steadier
gating). Testing empirically (user: "I might be wrong").
**Redesign of Obj6 wave to a clean same-time A/B (6-GPU cap = 3 slots):**
- 483 `llama8b-add-refine-01` = add-CONTROL (current refined recipe, ratio1, both LR 1e-3).
- 490 `llama8b-add-refine-treat-01` = add-TREATMENT (`obj6_add_refine_treat.yaml`:
  ratio 2.0, CI-fn LR 3e-4 / components 1e-3). [489 was 1st launch -> OOM, see lesson]
- 484 `llama8b-sub-refine-01` = sub-CONTROL (kept running; reference for a later sub-treat).
- CANCELLED to free slots: 485 (mult-refine), 486 (lastpos), 487 (obj7), 488 (obj8).
  Will relaunch these with the WINNING recipe after the A/B calls it.
Compare 483 vs 490 on target_recon/rounded, L0, n_alive, PGD AND nontarget_recon/total_l0
(the metric ratio2 most affects) + AB-plot component quality. Let them run for real signal.
Helper: `python ~/pd_scratch/compare_runs.py <run_id>...` (latest slow-eval table).
Monitor: bs82ypsht (fires on 483/484/490 completion).

### OPS LESSON: don't relaunch immediately after scancel
Launching job 489 right after `scancel 485` -> CUDA OOM: SLURM reassigned 485's 2 GPUs to
489 before 485's processes released their ~30 GiB, so they collided on the shared GPU.
Fix: wait for the cancelled job's GPU memory to drain (or just relaunch — 490 succeeded a
minute later). Symptom in log: "Process NNNN has 30.18 GiB memory in use" on your GPU.

### ROADMAP UPDATED (2026-06-13) — 3 changes
1. **Obj6 step count: "50% more" -> "TWICE as many"** = 10000->20000 steps (was 15000).
   -> Regenerated all obj6_*_refine configs at steps*2=20000. RESTARTED the A/B at 20k
   (cancelled 483/484/490 @ ~step1900, cleared partial dirs, relaunched):
   - 491 `llama8b-add-refine-01` add-CONTROL (20k)
   - 492 `llama8b-sub-refine-01` sub-CONTROL (20k)
   - 493 `llama8b-add-refine-treat-01` add-TREATMENT (ratio2 + CI-LR 3e-4, 20k)
   ~1.55 s/it -> ETA ~8.7h. Monitor bflsxg6u9. Midpoint trigger bna3ke7aw (run-id based,
   still valid). Drain-then-relaunch worked (no OOM this time).
2. **NEW Obj6-bis** — refine last-token addition to get LOWER L0 than the all-position
   version (Obj3 lastpos L0 7.87 > Obj2 all-pos 5.21 = the anomaly to fix), with equal/
   better recon. Lever intuition: last-token reconstructs only pos 4, so far fewer
   positions to satisfy -> can push impmin/sparsity much harder. May raise impmin base or
   peak; test runs allowed. Allowed to GIVE UP after a few attempts. [TODO after Obj6 wave]
3. **NEW Obj9** — make the addition decomposition work on **layer 18's ATTENTION** (not
   MLP). Decompose attention matrices (q/k/v/o_proj). [TODO; verify target module names &
   that Conv1D/Linear targets apply to Llama attention]. [last objective]

Pending order after A/B resolves: pick winning recipe -> relaunch mult-refine + lastpos-
refine (Obj6) + sub-treatment if warranted -> Obj6-bis -> Obj7 (addsub) -> Obj8 (addmult)
-> Obj9 (attention). Configs obj7_addsub.yaml / obj8_addmult.yaml ready (10k steps, C256).

### A/B RESULT — addition control vs treatment (both 20k steps, FINAL)
| metric | control (491, ratio1, CIlr1e-3) | treatment (493, ratio2, CIlr3e-4) |
|---|---|---|
| target_recon/rounded | **0.00372** | 0.00476 |
| total L0 | 6.51 | **6.10** |
| n_alive gate/up/down | 58/28/53 = **139** | 43/23/48 = **114** |
| PGD | **0.00376** | 0.00483 |
| nontarget_recon/rounded | 0.00318 | 0.00357 |
| nontarget L0 | 0.259 | **0.147** |
**Trade-off, not a sweep.** Treatment = sparser (-18% alive) + much more TARGETED
(nontarget L0 -43%), slightly worse recon (both excellent, both PGD<<0.1). Control = best
recon but DENSER; its n_alive ballooned 93->139 as impmin released, while treatment held
105->114. Mechanism: the Obj6 impmin-release improves recon but reactivates comps; ratio2
+ slower CI-LR counteract that, keeping the decomposition sparse+targeted through release.
- Midpoint (step 8k) treatment led on recon too; control overtook recon only in the final
  third (free impmin release). So the knobs trade end-recon for stability/sparsity/targeting.
DECISION PENDING on ci-thr 0.8 AB-plot component quality (jobs 496 control / 497 treatment
find_alive -> plots). For "more accurate components" (interp goal) sparser+targeted
(treatment) likely wins; for raw recon, control. sub-treatment (495) running as a 2nd data
point vs sub-control (recon 0.0075/L0 4.03/na92/nontgtL0 0.076).

### A/B VERDICT: TREATMENT WINS (ratio2 + CI-LR 3e-4) — adopt for all remaining objectives
ci-thr 0.8 AB plots (pos 4, the "=" answer token), components with CI>0.8:
- control (491): down41/gate48/up33 = **122** high-conf comps; many DIFFUSE near-uniform
  "always-on" squares (redundant, non-selective).
- treatment (493): down25/gate22/up15 = **62**; each CLEANLY STRUCTURED (anti-diagonal
  a+b=const bands, units-digit lattices, single-operand bands, magnitude blobs).
Treatment expresses addition with ~HALF the strongly-active comps, each more selective +
quieter on nontarget. For "more accurate components" (interp goal) treatment clearly wins;
recon cost negligible (0.00476 vs 0.00372, both PGD<<0.1). User hypothesis CONFIRMED.
Plots: <run>/figures/ab_heatmaps_add_thr08/.

### CURRENT WAVE (treatment recipe, 20k steps, 6 GPUs): monitor bij07zpo5
- 495 `llama8b-sub-refine-treat-01` (sub-treatment, 2nd confirmation vs sub-control).
- 498 `llama8b-mult-refine-01` (Obj6 mult, treatment recipe).
- 499 `llama8b-lastpos-refine-01` (Obj6 lastpos, treatment recipe; also = 1st Obj6-bis attempt).
TODO when done: reeval(40x96) + AB plots ci-thr 0.8 (lastpos pos4 only). For Obj6-bis,
check 499's L0 vs all-pos treatment (493) L0 6.10 — if 499 L0 < 6.10 w/ >= recon, Obj6-bis
solved; else raise impmin on lastpos (only pos4 to reconstruct -> lots of sparsity room).

### REMAINING after this wave
- Obj6 deliverables: AB plots ci-thr 0.8 for add(done)/sub/mult/lastpos treatment runs.
- Obj6-bis: lastpos L0 < all-pos L0 (499 is attempt 1).
- Obj7/8: regenerate obj7_addsub/obj8_addmult with treatment knobs (ratio2 + CI-LR 3e-4),
  C=256, launch. AB plots both ops ci-thr 0.8.
- Obj9: addition on layer-18 ATTENTION. Targets `model.layers.18.self_attn.{q,k,v,o}_proj`
  (all nn.Linear; q/o 4096->4096, k/v 4096->1024 GQA). Recipe = addition treatment +
  retarget self_attn. Pick C per matrix (start 128; k/v smaller).

### OBJ6 SECOND CONFIRMATION + remaining-task results (treatment recipe, 20k)
- **Subtraction** (495 vs sub-control 492): STRICT WIN — recon 0.00685<0.00750, n_alive
  88<92, PGD 0.00705<0.00774, nontgt_L0 0.023 vs 0.076 (-70%), L0 tie 4.03. Recipe
  generalizes with no trade-off.
- **Multiplication** refine (498) vs original Obj5 mult-02: recon 0.0089<0.0114, PGD &
  nontgt better, BUT DENSER (L0 9.61>8.29, n_alive 140>107). Impmin-release traded sparsity
  for recon (still within Obj5 spec: L0<10, 47/mat<<205).
- Obj6 ci-thr 0.8 AB plots DONE for all 4: add/sub/mult/lastpos (+ controls for add).
  In <run>/figures/ab_heatmaps_*_thr08/.

### OBJ6-BIS — diagnosis + attempt 1
lastpos-refine (499): recon 0.00377 (BEST) but **L0 8.49 > all-pos 6.10** = anomaly NOT
fixed (worse than even original lastpos 7.87). Cause: reconstructing only pos4 is easy, so
released impmin lets the optimizer over-fit pos4 with MORE comps. Fix = push impmin hard
(huge recon margin: 0.00377 vs 0.00476 bar).
**Attempt 1 = job 511 `llama8b-lastpos-obj6bis-01`** (`obj6bis_lastpos.yaml`): impmin base
5e-5->3e-4 (6x, = proven-sparse subtraction value), peak_mult 10->5 (peak 1.5e-3, final
3e-4). Target: L0<6.10 AND recon<=0.00476. If overshoots/undershoots, adjust base.

### LIVE (6 GPUs, monitor b94gz7w5q): 506 obj7-addsub, 507 obj8-addmult, 511 obj6bis-lastpos.
All 20k/~8.7h. Remaining: reeval+AB(0.8) for obj7(both ops)/obj8(both ops); judge obj6bis;
Obj9 (attention) still to launch. Commit winning configs at the end.

### OBJ7 add+sub (506) DONE ✅ — component SHARING confirmed
recon 0.00587, L0 6.67, n_alive 125 (gate38/up29/down58), PGD 0.00604, nontgt_L0 0.187.
**n_alive 125 < add(114)+sub(88)=202** → combined run SHARES ~38% of components across the
two tasks (roadmap intuition confirmed: below the sum). C=256 ample (42/mat << 205).
TODO: find_alive -> AB plots ci-thr 0.8 for BOTH ops (--op=+ and --op=-).

### OBJ9 attention (513) LAUNCHED + WORKS ✅
Addition on layer-18 self_attn.{q,k,v,o}_proj (C=128 each), full best recipe. Step-0 eval
sane: all 4 attn matrices decomposing, n_alive ~123-126/128 (full at init), recon 0.147
dropping, per-matrix PGD k_proj hardest (1.01, GQA 4096->1024) / o_proj easy (0.004).
1.44 s/it -> ETA ~8h. Attention decomposition assembles + trains cleanly (new target type
works out of the box). run_id llama8b-add-attn-01.

### LIVE (6 GPUs, monitor bf7myxw6v): 507 obj8-addmult, 511 obj6bis-lastpos, 513 obj9-attn.
obj7 find_alive DEFERRED (no free slot) -> run when obj8/obj6bis frees one. Then AB plots
obj7(+/-)/obj8(+/×), judge obj6bis L0, reeval obj9 + AB plots, commit decisions.

### OBJ8 add+mult (507) DONE ✅ — sharing also confirmed
recon 0.00813, L0 8.97, n_alive 168 (gate51/up46/down71), PGD 0.00861, nontgt_L0 0.250.
**n_alive 168 < add(114)+mult(140)=254** → shares ~34% (less than add+sub's 38%; add & mult
are more different ops). C=256 fine (56/mat << 205). find_alive 514(obj7)/515(obj8) running
-> AB plots ci-thr 0.8 both ops each.

### OBJ6-BIS — trade-off mapped, bracketing for the corner
Target: lastpos L0 < all-pos 6.10 AND recon <= 0.00476. Trade-off curve (impmin base):
- 5e-5 (lastpos-refine): L0 8.49, recon 0.00377  (recon great, L0 too high)
- 3e-4 (attempt1, 511):  L0 3.40, recon 0.00737  (L0 great, recon over bar) -> OVERSHOT
Interp suggests at recon=0.00476, L0~7; at L0=6.10, recon~0.0055 — corner may be tight.
Confirmed aux recon losses (Unmasked/Stochastic) USE ctx.reconstruction_loss = the
last_pos variant for these runs, so they respect last_token (not fighting sparsity); it's
a genuine impmin trade-off. Cancelled attempt2 (5e-4, wrong direction). Launched 2 brackets:
- 519 obj6bis-03 impmin 1.2e-4
- 520 obj6bis-04 impmin 1.8e-4
Pick the one (if any) with L0<6.10 AND recon<=0.00476; else report the trade-off (L0
anomaly IS fixable — L0 well below 6.10 — at a small recon cost) and move on (give-up allowed).
Note: L0 3.40/recon 0.00737 already shows lastpos CAN be far sparser than all-pos; strict
recon-corner is the open question.

### OBJ9 addition-on-ATTENTION (513) DONE ✅ — works, very sparse
recon **0.00136** (lower than MLP!), L0 4.12, n_alive 58 total, PGD 0.00137, nontgt_L0 0.069.
Per-matrix alive: q_proj 6, k_proj 3, v_proj 27, o_proj 22 (L0 q0.84/k0.80/v1.11/o1.36).
FINDING: attention plays a SMALL, sparse role in addition — q/k nearly trivial (3-6 comps),
v/o carry the little work there is. Consistent with addition being an MLP computation.
C=128 oversized (max 27 alive) but harmless. find_alive 521 -> AB plot --op=+ ci-thr0.8.
Attention decomposition works out of the box (no code changes needed).

### OBJ9 AB plot (ci-thr 0.8, pos4): high-conf attention activity concentrates in O_PROJ
(structured a/b diagonals+lattices+bands); q/k/v contribute few high-conf comps. Attention's
addition role lives mostly in the output projection. Obj9 fully DONE (run + AB plots).

## ===== ROADMAP STATUS: Obj1-9 essentially COMPLETE (2026-06-15) =====
Only OPEN item: Obj6-bis recon-corner (brackets 519=1.2e-4 / 520=1.8e-4 running, ~6.5h) —
does lastpos reach L0<6.10 AND recon<=0.00476? L0 anomaly already shown fixable.
THEN: consolidated commit decision (surface to user; don't overwrite Obj2/4/5 configs).
Untracked repo file to commit-or-keep: prompts/addition_multiplication_1-100.txt (Obj8).

### ADDMULT RETRY (user idea, 2026-06-15): peak-high / release-floor-LOW + frequent ckpts
Hypothesis: high impmin MERGES components; need high peak for sparsity then release floor
MUCH lower so merged subcomps re-separate/re-activate; optimal decomp may be mid-training.
**job 522 `llama8b-addmult-02`** (`obj8_addmult_v2.yaml`):
- impmin: peak 5e-4 (SAME, base 5e-6 x peak_mult 100), release floor to 5e-6 (10x lower
  than prev 5e-5) over steps 2k->10k, hold low 10k->20k.
- save_every=1000, keep_last_n=20 (full trajectory, 16GB each, 11T free) -> pick optimal ckpt.
- **ci_scaled_weight_decay 0.2->0.0** (LEARNED TRICK: ci_wd shrinks low-CI comps' weights by
  (1-maxCI)/step -> a suppressed comp's weights decay away and CAN'T re-activate; disabling
  lets them retain weights to come back).
- C 256->384 (re-activation RAISES n_alive; need headroom above prev 168).
- keep treatment (ratio2, CI-LR3e-4). Step-0 OK: n_alive 1151/1152, recon 0.32, 1.54 s/it.
WRAP: after done, plot n_alive(t) + recon(t) to find re-activation onset; AB-plot a few
candidate ckpts (mid-release vs end) to see if components un-merge. Monitor bzt5vgkcl.

### LASTPOS AB PLOTS (user req): pipeline bd4h8fbdr — waits for 519/520, find_alive +
plot_ab_heatmaps --op=+ --pos4 --ci-thr0.8 for obj6bis-01/03/04 -> figures/ab_lastpos_thr08/.
(add-lastpos-01 + lastpos-refine-01 already have ab_heatmaps_add_thr08/position_04.png.)

### OBJ6-BIS VERDICT (4-point curve mapped): L0 anomaly FIXED; strict recon-corner not reachable
Final: 5e-5 -> L0 8.49/r0.00377 | 1.2e-4 -> 5.65/0.00552 | 1.8e-4 -> 4.54/0.00572 | 3e-4 -> 3.40/0.00737.
Wherever L0<6.10, recon>=0.0055 (>all-pos 0.00476). BEST = obj6bis-03 (1.2e-4): L0 5.65,
n_alive 76 (both << all-pos 6.10/114) -> the anomaly (lastpos L0 > all-pos) is FIXED.
CAVEAT: lastpos recon is pos-4-ONLY (hardest token); all-pos 0.00476 is a 5-position AVG
(diluted by easy tokens) -> unfair. Fair test = all-pos run's pos-4-only recon vs lastpos
0.0055 (TODO: 1-GPU eval of 493 @ pos4). Likely flips to lastpos-favorable. Deliverable run
= llama8b-lastpos-obj6bis-03.
