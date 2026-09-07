# BFM program handoff — read this first

**Status: ACTIVE. Last updated 2026-09-07.**

⚠ **If you read only two things: [§10.0](#10-why-bfm-has-not-been-learned) — **BFM and the kill are
INDEPENDENT on this instrument**, measured in both directions, which supersedes both §10's own
2026-09-06 diagnosis and the lever-by-lever framing of §3/§4 — and B1's verdict in §9 (4 → 45 kills,
the program's largest gain).**

⚠ **Live as of 2026-09-07: NOTHING IS RUNNING.** All twelve Phase-A/E boxes and **F2–F8** are closed, and
the **c97 pair is judged** — the 45-obs treatment **lost** to its own byte-identical 41-obs control (4
kills vs 25) and never learned to read the appended columns (encoder-norm ranks 42/44/41/39 of 45), so
**c97's hypothesis is UNTESTED, not refuted**. The result that matters came from the **control**: the
**≥55° wall is broken** (asp55 8/20 vs the same-invocation c96 anchor's 0/20) ⚠ by **redistributing**
kills up the hemisphere (45 → 25 total), one rung only, and with **zero detectable BFM**. The **D1
detector repair is landed, audited (F7) and now re-derived on policies (F8)**: a merge/head-on D1 count
is admissible as evidence of **absence** only, and F8 retracted the program's most-cited policy-side
reading (see §10.0 — *"c95 = 0.00" was one cell*). **The first unchecked box in §6 is an open decision
for the user: which checkpoint is the baseline, c96 or c97-ctl.**

This is the living work order for the "teach the RL policy real BFM" program. If you are an agent
picking this conversation up after a context compaction, **read this whole file before doing
anything else**, then continue at the first unchecked box in [§6 Checklist](#6-checklist).

Companion long-form notes live in the session memory directory
(`~/.claude/projects/-home-anthony-nguyen-aca-astrum/memory/`), indexed by `MEMORY.md`. This file is
the *plan and progress*; memory holds the *evidence*. Where they disagree, this file is newer.

---

## 1. The goal (the user's standing intent)

Train the TD-MPC2 agent in the ASTRUM WVR dogfight environment to learn **out-of-plane maneuvers,
displacement rolls, and turn-circle geometry** and to use them to kill with the cannon. Pure 1v1
from a **neutral head-on** start (explicitly *not* the aspect < 90° advantageous setups that the
first ~90 curricula fixated on), optionally with a slight altitude edge for energy.

The user cares about **both halves**: the agent must learn *how* to fly the maneuver **and** *when*
to commit to it. The engagement should look like real WVR combat, not like a scoring exploit.

Governing decisions already made by the user and still in force:

- **"Fix it and re-baseline"** — the c01→c93 score chain is void; the T3 opponent was defective.
- **"Neutral head-on vs the fixed T3"** — the benchmark geometry.
- **"Go to the lateral separation"** — the head-on spawns get lateral offset (turning room).
- Making the reward's WEZ the real WEZ was **endorsed** — **now DONE (B1) and it produced the largest
  gain in the program's history, 4 → 45 kills. See §9.**

---

## 2. Where the program actually stands

`c87` is the best chase policy ever produced (32 kills / 0 losses on the *old*, defective
benchmark). Against the **fixed** T3 it scores 2 kills / 29 decks, so ~94% of its headline was an
artifact of the broken opponent. `c95` (lead-turn prior) is the best post-re-baseline run: 10 kills
/ 1 deck on the chase, and it moved sub-8 m lay dwell from 0.01 s → 0.10 s — but **the policy alone
still converts nothing at the merge.**

Twelve reward-shaping curricula in a row produced nothing. The only two levers that ever worked were
**representation** changes (c86 corrected gun lay, c87 arrival-geometry obs). The working prior is
therefore: *ask what the policy can SEE before asking what the reward PAYS.*

---

## 3. The six refuted mechanisms (do not re-run these)

A 13-agent adversarial probe on 2026-09-05 tested six candidate explanations for "why no BFM".
**All six were refuted**, three of them by direct sim execution rather than argument:

| Mechanism | Verdict |
|---|---|
| Coloured (OU/pink) exploration noise | ⚠ **MIS-FILED — the correct status is UNTESTED. See §10.1.** The original entry read: "Dead. Noise injection lives inside `plan()`, and `cfg.mpc=False` in every checkpoint, so a noise flag is a silent no-op." That says **the experiment never ran**, which is not a refutation. `OUNoise`/`PinkNoise` are dead code in every run to date. The secondary argument (`abs(elev) > 0.5` for 29.7 s) is the *command-side* error D4 made twice — sustained `elev` is a g command and says nothing about a sustained out-of-plane ROLL; the load-gated measurement says max χ hold 5.40 s, D1 = D2 = 0.00. |
| MPC on for collection / longer `cfg.horizon` | ⚠ **Refutes the FIX, CONFIRMS the mechanism. See §10.4.** `config.py:69 rho = 0.5` caps *trained* latent-rollout depth at ~10 control steps (~417 ms) for **every** `H`, so raising `H` buys no depth — correct, and it establishes that the stack **cannot represent a multi-second commitment**, which is a live defect, not a closed one. `rho` itself has never been varied. |
| Append CPA / turn-circle / maneuver-plane obs features | **Dead.** An MLP recovers all of them from the existing 41 floats at R² 0.82–1.00. The information is already there. |
| Stage the merge lower (more vertical room) | **Dead, by execution.** Block at 5000 ft → 6/6 hard decks at 15.5–17.9 s, vs 4/6 at 53.7–75.0 s. There is no vertical room left to displace *into*. |
| Re-form the γ=1 PBRS differencing | **Dead.** Real but ~5% delay tax on something worth 20× more; the "fix" shocks reward scale 4–20×. |
| Advantage-filtered BC / de-ratchet `K_LEAD` / rescale `ALPHA_REF` | **Dead.** `ALPHA_REF` is a monotone bijective rescale of 1 of 41 columns, absorbed by the first Linear. |

**The decisive negative:** `HighYoYo` already scores **+0.01 (chase) / −2.94 (merge)** against the
in-plane tracker over a whole episode. ~~Out-of-plane maneuvering is already reward-free. There is no
prohibition to lift~~ ⚠ **MISREAD — see §10.2. −2.94 is not "free", it is PRICED**: four dense terms
(aim, closing-range, stern-conversion aspect, energy) all go negative during a correct maneuver, and
`reward.py:157` states outright that "the K_CLOSE/K_AIM PBRS go NEGATIVE on the overshoot". The valid
half of the original conclusion survives and is strengthened: *do not add a reward constant to pay for
maneuvering* — it cannot outrun four charges. **Stop charging instead.**

⚠ **Read §10 before treating any row above as closed.** Three of the six entries have been re-graded:
two mis-filed as refutations of mechanisms when they refute only a fix or never ran, and the "decisive
negative" inverted. The other three (obs features, staging lower, `ALPHA_REF`) stand.

---

## 4. What actually survived — five real defects

These are all verified in source, not inferred.

### D-A. The demonstrations have never been the maneuver (`train_parallel.py`)

> **VERDICT (A2, §9): the heading is right but stronger than written, and none of the three numbered
> defects is the reason.** The sticks do not fly the maneuver *even when run continuously with every
> defect neutralised* — D1 = D2 = 0.00 for all five arms at all thresholds. So the plumbing bugs
> below are real but **not levers**, and no behavioural change was shipped from them. Read 1/2/3 as
> source-accurate archaeology, not as a work order. **2 is outright refuted** (rudder 0.0 is
> correct); **3 is refuted as a mechanism** (the yo-yos never reach the branch that carries the
> constant).

1. **The per-episode coin is `leadturn`-only.** `train_parallel.py:1006-1013`: only
   `merge_override_mode == "leadturn"` reads the per-episode `lt_episode_on`. `highyoyo`,
   `lowyoyo` and `egg` draw a **fresh coin every control step** (24 Hz). Since `_highyoyo_stick`
   is stateless, the "demonstration" is a quarter-second of script, a quarter-second of policy,
   alternating. You cannot demonstrate a multi-second maneuver that way. The existing comment
   argues memoryless sticks "degrade gracefully" — that is wrong for a *closure-phased* stick,
   because the policy's interleaved ticks change the closure the stick keys its phase on.
2. **Rudder is nailed to zero** in all four sticks: `:197`, `:249`, `:305`, `:493` each return
   `[ail, elev, 0.0, thr]`. None of them is a coordinated roll.
3. **Throttle constants are in the wrong units.** Agent space is `[-1,1]`; native throttle is
   `[0,1]`; `map_tdmpc_action` gives `native = 0.5·(a+1)` (`envs.py:103-114`). So
   `_HY_THR_UP = 0.70` commands native **0.85**, and `_LY_SETTLE_THR = 0.55` commands native
   **0.775**. The comments say "ease throttle to help bleed the surplus closure" — it doesn't.
   **The energy-bleed half of every yo-yo has been running at roughly double the intended power.
   A high yo-yo that cannot bleed is a zoom.** Same bug in `_leadturn_stick:492`
   (`0.5 + ΔV/40` is agent space, so the intended "half throttle" is native 0.75).
4. **The demos are flushed before the run ends.** `merge_override_anneal = merge_override_steps /
   workers` (`:3432`) and `steps_in_stage` is per-worker, so c95's prior went off at 3M of 5M.
   With `buffer_size = 1_000_000` (`config.py:86`), **the final checkpoint's replay buffer held
   zero demo transitions.**

### D-B. There is no channel that copies a maneuver into the deployed policy

`update_pi` (`agent.py:241-259`) is exactly
`((entropy_coef·log_pis − scale(qs)).mean(1)·rho).mean()` and takes only `zs`. `buffer.py:42-47`
allocates six rings with **no demo label**. There is no BC/AWR/demo term anywhere. Combined with
`cfg.mpc=False` (the actor *is* the deployed controller), a spliced demonstration provably trains
only encoder/dynamics/reward/Q — **the maneuver shape has never been copied to the policy.**

### D-C. Every jet has been flying at ~half its available g — **REFUTED (A3, §9)**

> **VERDICT: false.** Both candidate fixes make the jet turn *worse* than stock (sustained g −0.3 to
> −0.9, radius +15–42%), and both settle at lower alpha, so the stock loop is not suppressing a pull
> it could otherwise deliver. The "4.7–5.8 g" figure was the **worst corner of the envelope**: the
> same stock jet sustains 7.84 g at 5 kft / 260 m/s and peaks at 10.76. Keep the reading below for
> the source detail, but **do not act on it.**

`/opt/jsbsim/share/jsbsim/aircraft/f16/f16.xml:586-603` (inside the `dogfight-train` container)
wires `fcs/elevator-pid-trigger` — which **defaults to 1** and drops to 0 only below `vc 5 kts` — as
the `<trigger>` of `fcs/g-load-pid`. A triggered JSBSim `FGPID` **does not advance its integrator**.
So the g-load loop is **proportional-only (kp 0.3) in all normal flight; ki 0.025 never integrates**,
leaving a permanent droop between commanded and achieved Nz. Sustained n measures **4.7–5.8 against
a lift limit near 11**.

This is the real cause of the "5.8 g / α ceiling 16.7°" figures previously filed as *airframe* facts.

⚠ It is a **third-party asset outside the repo** — an in-place edit is invisible to git. Ship a
**named aircraft variant**, and note that changing it voids c01–c95 comparability exactly as the T3
fix did.

### D-D. We cannot currently detect BFM at all — **CLOSED (A1, §9)**

Two probe agents reached **opposite** conclusions about roll coherence purely from statistic choice
(one-sided `ail > 0.5` → "max run 18 steps"; sign-agnostic `abs(ail) > 0.5` without sign-constancy →
"held 3.54 s"). Both are wrong for the question. Until a detector exists, **a run that scores better
is indistinguishable from a run that learned BFM** — which is precisely the failure the user asked
to avoid.

### D-E. The reward pays for shots that physically cannot hit — **FIXED (B1, §9); run pending**

> **CORROBORATED INDEPENDENTLY (A2, §9).** Scripted sticks with no checkpoint involved reach a
> **1.9 m** lay (in-plane tracker) and a **3.9 m** lay (lead turn) and record **zero** lethal ticks,
> because those lays land outside 600 m. Lay accuracy is not the wall — **range at the moment of
> solution** is. Four of the five §4 defects are now refuted or closed; this is the one that survived
> contact with measurement, and it survived twice.

`range_factor = (R_EFF − rng)/R_EFF` with `R_EFF = 1200` (`geometry.py:229`, `constants.py:48`), but
measured p_hit/round against a maneuvering bandit is 0.98@300 m, 0.48@500, 0.20@600, 0.06@700,
**0.000@1200**. The lethal envelope is ~**600 m**. The greedy policy fires at a median **1519 m**
and 0% of its firing ticks fall inside 600 m. The mag dump is not an exploration artifact and not a
waste-penalty calibration issue — it is a **range** problem, and the reward is paying for it.

### Bonus: five measurement bugs that corrupt the record

- Four diagnostics report `hypot(los_rate_az, los_rate_el)`, which is **0.70× the mean and 0.32× the
  p10** of the physical `abs(v_perp)/rng`. Sites: `_diag_dwell_gate_tmp.py:97`,
  `_diag_headon_alt_tmp.py:114`, `_diag_merge_convert_tmp.py:312`, `_diag_trigger_tmp.py:81`.
  **Every LOS-rate number in this program's history is a lower bound**, including the 18.6°/s vs
  59°/s arrival measurement that set `LOS_RATE_REF` itself.
- `_diag_trigger_tmp.py:69` tests `a[4] >= 0.5`; the contract is `>= 0`. Every historical
  trigger-down figure **undercounts**.

### Not defects (previously recorded as such — corrected)

- **`obs[31]/[32]` is NOT a bug.** It is a lossless non-orthogonal basis (reconstructs to 2.4e-8
  median relative error). The proposed "fix" has a `1/lx` pole and saturates its own ±2 clip on
  6.5% of frames inside 600 m. **Do not touch the obs.**
- **MPC was not measured harmful.** Pooled over all five historical comparisons: greedy 24/516 vs
  MPC 14/356, Fisher **p = 0.74** — no detectable effect.

---

## 5. The one thing the analysis got wrong, and the user got right

A synthesis pass argued the gun kill is geometrically hopeless because turn diameter (2R ≈ 1960 m)
dwarfs the 600 m gun radius. **That conflates diameter with reachable separation** and is wrong.

The sound argument is **symmetry**: two identical jets pulling identical max-rate g *in the same
plane* hold their angular separation forever, regardless of how well either flies — a draw by
symmetry, not by radius. On a shared circle, being inside 600 m means being within ~36° (R≈980 m) to
~44° (R≈800 m) of arc: tight, but reachable.

Which is exactly **why out-of-plane BFM exists in the real world** — it is the only way to break the
tie. The user's intuition is the correct one and this finding supports it.

---

## 6. Checklist

Ordered **cheapest-first**. Steps 1–4 are zero-GPU and can all be done without launching a run.
Do **one edit per run** thereafter ([[astrum-run-economics]]: a 5M-step run is ~40–60 min).

### Phase A — free, no GPU, no training

- [x] **A1. Build the BFM detector** → `crates/astrum-env/python/astrum_dogfight/bfm_detect.py`.
      Pure function of a per-tick trace. Spec in memory `bfm-detector-spec.md`; summarized in §7.
      Without this, nothing downstream is measurable. **Done — see §9.**
- [x] **A2. Measure the demo sticks' own ceiling** with the A1 detector before spending any GPU
      chasing them. Harness: `crates/astrum-env/python/_diag_stick_ceiling.py`.
      **Done — the library is HOLLOW. D-A/1/2/3 are all real in source and NONE is a lever; no
      behavioural change was shipped to `train_parallel.py`. See §9.**
- [x] **A3. Measure the FCS integrator** (D-C) — `make jsbsim-variants` + `_diag_turnperf.py
      --model {f16,f16nopid,f16kp10}`. **Done — D-C REFUTED: the stock FCS out-turns both variants
      everywhere. Do not ship an FCS change. See §9.**
- [x] **A4. Fix the five measurement bugs** (four `hypot` LOS-rate sites + the `a[4] >= 0.5`
      threshold). **Done — fixed at the root by adding `RelGeom.los_rate`; the trigger bug was in
      five files, not one. See §9.**

### Phase B — one run, one edit

- [x] **B1. Make the reward's WEZ the real WEZ** (D-E): replace `range_factor`'s linear
      `(R_EFF − rng)/R_EFF` ramp with the sim's own measured p_hit curve. Still the single largest
      measured defect. User has already endorsed this lever.
      Judge with the standard greedy verdict: `--aspects 0,30,45,55,60,75 --episodes 20
      --jitter 0.5 --modes greedy`, **plus** the A1 detector's D5 ratio.
      **DONE, both halves of the judging rule. Largest measured gain in the program's history: 4 → 45
      kills (>11×), losses unchanged at 1, one edit.** Run `runs/curriculum-96/`, warm from c95 on
      c95's line verbatim. c95 re-measured in the SAME format for the delta (no saved baseline
      existed). The ≥55° wall is UNTOUCHED — and c96 spends MORE rounds there, so the trigger was
      never its binding constraint. **D5 half: c96 kills 6/6 at aspect 30 with D1 = D2 = D3 = 0.00 ⇒
      the gain bought ZERO out-of-plane maneuvering; it is an aim fix and it leaves the user's actual
      goal untouched.** See §9.
- [x] **B1a. Correct D4 and build the policy-side BFM harness** — unplanned, found while preparing
      B1's verdict: D4 measured the roll-**RATE** command, and `_bfm_verdict_tmp.py` did not exist, so
      the "plus the D5 ratio" half of B1's judging rule was unrunnable. Both shipped; the first policy
      read (c95: D1 = D2 = D3 = 0.00, `|χ|` median 12.2°) is in §9.

### Phase C — was gated on A2's result

- [x] ~~**C1. Actor-side BC**~~ — **CLOSED by A2's gate, which returned NO.** The gate was "only if
      A2 shows the corrected stick is worth copying". It isn't: every arm scores D1 = D2 = 0.00 with
      zero lethal ticks over 40 episodes, and the sensitivity sweep rules out a threshold artifact.
      **There is nothing to BC-copy.** The λ-parity and closure-gating design notes are preserved in
      git history if a demonstrator ever exists that actually flies the maneuver — the blocker was
      never the BC mechanism, it is the absence of a demonstration.

### Phase D — the board (authored 2026-09-06)

Authored by a 26-agent orchestrated pass: four independent sweeps of the record (this document, the
memory directory, **source comments/docstrings**, and the config negative-space) produced a 277-entry
ledger of recorded verdicts; three proposers worked against that ledger; every candidate then faced two
adversarial skeptics, one attacking on the record and one on the executed path. **9 candidates, 5 closed
with citations, 4 surviving** — and in every surviving case the form that survived is a skeptic's
salvage, not the original proposal.

⚠ **Two amendments I made to the board after checking its own load-bearing claims** (it was right to
correct §10.2/§10.3, and then `_cf_out.txt` turned out to correct it further):
1. **The §10.5 ordering mandate is CELL-SPECIFIC, so D2 is no longer blocked on D1.** On the chase the
   measured return already favours the max-g pull by +47 to +57/episode and the policy declines anyway
   (§10.2). There is no sign to fix there ⇒ **D2's exploration measurements are unconfounded on the
   chase today.** The board's own sequencing line ("D2's run only after D1b") applies to the merge only.
   This makes D2 the highest-value *runnable* item, not the third.
2. **D0.1 is superseded by the corrected §10.2/§10.3** — I applied both corrections directly to §10
   rather than leaving them as a to-do, and went further than the board did: the charge is **two** terms
   (`K_CLOSE` −82, `K_LEAD` −32), `energy` measured **positive** in all three arms, and `aspect` is
   sign-inconsistent. D0's remaining items (the 41.7 ms constant, copying the two single-location
   refutations into §9) are still open and still worth the ten minutes.

Phase A–C are closed (A1–A4, B1, B1a shipped; C1 returned a negative). This section is the live work order and replaces §10.6's closing prose as the plan of record. **Continue at the first unchecked box.**

⚠ **READ THIS FIRST — board state as of 2026-09-06, after the free tier ran and was adversarially
verified.** Five of the seven boxes are closed: **D0** done, **D1a FAIL ⇒ D1b closed unrun**, **D4 FAIL
(the mechanism is vacuous on the run cell)**, **D5 INCONCLUSIVE but it surfaced two new source-level
critic defects**, **D6a FAIL ⇒ D6b closed unrun on arithmetic**. **D3a is PARTIAL** — its measured half
stands, its named band does not. ⛔ **UPDATED 2026-09-06 — THE ENTIRE BOARD IS NOW CLOSED. Phase D 7/7,
Phase E 5/5, ZERO live boxes and ZERO pre-authored runs.** The last two closed the same day: **D2d NEGATIVE**
(OU ties the incumbent at p = 1.0000, `D1 = D2 = 0.00` on all three arms ⇒ exploration is not the limiter) and
**D3b split** (the merge kill IS reachable — 5/8 seeds where all three controls deal 0.0 HP — but with **zero**
loaded D1 / clean D2, so the pre-registered negative stands on the out-of-plane half). Anything from here must
be authored fresh, and the three closures point one way: **§10 needs rewriting around the finding that killing
and BFM are close to independent in this environment** (D3b's third instance), not around another lever. The pre-registered **contingency has FIRED** — see the
"⚠ THE CONTINGENCY FIRED" subsection at the end of this section for what that changes, and §9's
"The free tier" entry for the numbers. Eight of the ten verifiers found the reported statistic unsound, so
**read the corrected verdicts in §9, not any box's original prose.**

Nine candidates were proposed and each was put through two independent adversarial passes. Four were closed outright with citations and are listed at the bottom as **checked and closed** — do not re-propose them. Five survive, and in **every** case the surviving form is a skeptic's salvage, not the original proposal: each original candidate's judging rule was defective in a way that would have manufactured or destroyed its own result. Where I preferred the salvage, it is said so in the box.

Be honest about the expected value of this board. The strongest stop-charging form (D1) has a gate predicate that the record's own paired data says is *anti*-correlated with lethality — c95 has the longer loaded chi hold at aspect 55 and zero lethal ticks, c96 has the shorter hold and 13 — so it may well die on its own free pre-check. Every discovery lever (D5, D6) is uninterpretable in both directions until D1 moves the sign (§10.5). The item with the highest ceiling (D3, true-sim winnability) is the one whose most likely outcome is *"the target may not be reachable as posed"*. So the realistic best outcome of Phase D is **closing branches decisively for near-zero GPU cost**, not a kill-count gain. Four of the six boxes below are free, and three of those can kill their own lever before any run is launched. That is where the value is concentrated — spend it there first.

Two standing rules for every box: judge only the out-of-band greedy verdict (`--jitter 0.5`, aspect 30 included, `collected ≈ 5M` verified in the header, FINAL checkpoint, `run.pt` snapshotted), and run the D6 control set (c96 + c95 + c87) **inside the same invocation** — both BFM claims ever made without one were wrong.

---

#### D0 — record repairs (not an experiment; do it first, two boxes below cite it)

- [x] **change:** three corrections to this document, ~10 minutes, no measurement. — **DONE 2026-09-06.**
  (1) was already applied when the board was authored (see §10.3's ⚠ banner). (2) the 41.7 ms constant now
  opens §7 with a seconds→steps table and the corrected γ/world-model arithmetic. (3) both refutations are
  in §9 under "D0 — two single-location refutations" — **and (a) needed reconciling, not copying:**
  `_refute_noise_band.py` measured the arm as MPPI's *final action draw* on the dead `plan()` path,
  post-elite-mean, post-clamp and **post-squash**, and reported the std of the *absolute* command; D2c
  measured the deviation on the live pi path pre-squash. Both stand; different injection sites.
  1. `docs/bfm-program-handoff.md` §10.3 mis-cites `reward.py:332` as a policy-invariance breaker. `band_r` at `crates/astrum-env/python/astrum_dogfight/reward.py:345` is `K_BAND*(_band_phi(me.airspeed) - _band_phi(prev_me.airspeed))` — an unconditional difference of a state function, which telescopes exactly. "ONE-SIDED" at :332 describes the *shape* of Φ (flat above corner), not a gate. The actual non-invariant sites are `reward.py:316`, `:371` (the two `engaged` gates) and the `K_LEAD` ratchet at `:390`; the first two are now **measured inert** (see closed item C-3).
  2. A control step is **41.7 ms** (`action_repeat 5` / 120 Hz sim, `config.py:20`), not 208 ms. Three separate Phase-D candidates made the same 5× error and mis-sized their own interventions by 5×. Put the number in §7 next to the γ arithmetic.
  3. Copy into §9 the two refutations that currently live in exactly one place: the executed-action band reading (`_refute_noise_band.py`: 1.5 s window-mean aileron **0.360 white → 0.322 OU**, i.e. *down*), and the c62 pure-outcome catastrophe (only in the `reward.py:106-110` docstring). Single-location refutations have now caused three separate re-proposals.
- **cost:** free.

---

#### D1 — Commitment-gated K_LEAD relief, in the completion-conditional form (the surviving instance of §10.6's OPEN lever)

Preferred in **skeptic 1's salvaged form**, not as proposed. The original waived the out-of-zone `K_LEAD` toll per-tick on attitude+load; both skeptics showed that predicate is cleared more cheaply by a loaded deck-recovery pull-up or by c87's sawing-χ spiral than by a displacement roll, is monotone non-decreasing in dwell (a *dwell* farm, not a cycling farm), and — fatally — is bit-identical to the metric the run would be judged on. The salvage rebates at excursion **close**, gated on completion.

- [x] **D1a (free, and it is the gate): measure the predicate before touching the reward.** — **DONE
  2026-09-06: FAIL, both verifiers upheld it, and the contingency below has FIRED (§9).** Both
  pre-registered FAIL conditions fire on the chase at **all nine** (χ, nz) settings; c87 is the highest-
  clearing arm of all five; and in the currency D1b pays in (per **excursion**) c87 beats the yo-yo arm on
  **both** cells — 9.42 vs 2.33/ep headon, 6.00 vs 0.00 chase. **The detector-gated-waiver family is dead
  as an attitude-and-load predicate.** Two instrument traps banked in §9 that outlive the box: tightening
  the load bar **fragments** holds rather than purifying them, and χ is ill-conditioned near boresight
  (`|los_perp| = sin(ata)` vs a 5° validity bar) exactly where the policy arms' clearances live.
  Instrument the tick fraction of `|chi| > CHI_ROOM_DEG ∧ nz > NZ_LOADED ∧ ¬in_zone` (`bfm_detect.py:68`, `:74`, `out_of_plane_angle` at `:127-180`; `in_zone` per `reward.py:388`) on five arms: A2's `inplane` (LagPursuitTracker) and `highyoyo` from `bfm.py`, plus recorded greedy rollouts of c96, c95 and c87.
  **Prediction that can come out the wrong way:** the record predicts this **fails** — `inplane` runs nz p50 5.26 and already scores 0.25 D1/ep once only the χ bar drops to 30° (`handoff:374`, `:393-395`), and c87 holds 39.76 s of *loaded* constant-sign bank while decking 5/5 (`:803-814`). Pass requires `inplane` **and** c87 both near zero **and** the out-of-plane arms materially above. If `inplane ≥ highyoyo`, or if c87 clears the gate as often as c95/c96, **the whole detector-gated-waiver family is dead as an attitude-and-load predicate** — stop, record it, and go to D3/D4 (see the contingency note at the end).
  **cost:** free, ~1-2 h CPU.

- [x] ~~**D1b (one run, only if D1a passes): completion-conditional rebate.**~~ — **CLOSED UNRUN
  2026-09-06. D1a failed its gate, so do not build the accumulator.** Kept below only because its judging
  note ("holds rising while `chi_hold_dalt_m` worsens = the c95 unloaded-spiral signature") turned out to
  describe exactly what D1a measured: on the headon the gate ranks c95's −14,807 m descending spiral 2nd
  and the `highyoyo` +2045 m zoom-away egress 1st. One residual worth keeping: every D1a gate fraction is
  an **upper bound** on what a waiver could pay, because `reward.py:390` makes the out-of-zone `K_LEAD`
  term one-sided.
  **change:** accumulate the out-of-zone `K_LEAD` charge *inside* an excursion and emit the rebate at excursion **close**, gated on D1's full clause set — χ returns under `CHI_RETURN_DEG=30` within `CHI_RETURN_S=4.0` **and** aspect gained ≥ `ASPECT_GAIN_DEG=15` (`bfm_detect.py:316-360`). `outcome_mode` only. Plumbing follows the shipped accumulator pattern exactly: declare at `env.py:141`/`:149`, zero in `reset()` `:179-182`, pass positionally into `compute` `:200-211`, read back `:213-215`, surface via `info_out` `:258-268`. This costs the "stateless" property and is the price of not being monotone in dwell — a departure, a deck-recovery pull-up, or a spiral that never returns and never buys angles collects **nothing**.
  Two mandatory extras: (i) **exclude the rehearsal anchor** — pass the waiver per-stage, not through the shared `env_kwargs` at `train_parallel.py:3476`/`:4795`, using the exclusion path that already exists for `me_dalt_ft`/`merge_alt_ft`/`merge_lateral_m` at `:3236-3260`, or the waiver bites the >50°-aspect reposition legs of the ≤55° finish it exists to preserve. (ii) **decouple the reward gate from the verdict** — set the reward's bar at `CHI_OUT=60°` while the detector reports holds at `CHI_ROOM=45°`.
  **prediction:** offline, rescoring `_refute_diff.py --scenario merge` as PATCH-C, `C_highyoyo`'s incentive shift ≥ +5 while the **c87-spiral arm** (below) is ≤ 0.2× of it, flipping the sign of the recorded −2.94 HighYoYo deficit. Online, one 5M warm run from c96 with `--outcome-reward`: aspect-55 `tLeth` (sub-8 m lay **and** rng ≤ 600 m) rises from 13 ticks and at least one aspect-55 kill appears, with aspect-30 kills holding ≥ 5/6.
  **judging + negative control:** free pre-checks first — replay recorded c96 rollouts through patched vs shipped `compute` and report `|ΔJ|` **as a fraction of the baseline `|J| ≈ 560`** (B1's accepted precedent was 1.14×; anything past ~1.2× is the c88 shock class, do not launch). **The negative control is c87, replayed** — 39.76 s of loaded constant-sign bank, 2.45 s chi hold, −2642 m per hold, 5/5 decks, zero lethal ticks: a loaded nose-off descending break. If c87's rebate matches or exceeds the yo-yo's, the gate is cleared by the program's dominant failure mode and the lever is dead. **`D_straight` is not a control** — it is `[0.0,-0.05,0.0,1.0]` at nz≈1 (`_refute_diff.py:139-140`), excluded by the load clause arithmetically, and cannot fail. Also kills it: judged holds rising while `chi_hold_dalt_m` worsens and `tLeth` stays flat (the c95 unloaded-spiral signature); aspect-30 kills < 4/6; gated-tick fraction < 0.5% (too rare to matter).
  **cost:** ~half a day for the accumulator plumbing, one 5M run (~40-60 min).

---

#### D2 — Measure the actor's own exploration amplitude on the executed path, then land the noise plumbing as a correctness fix (free)

Preferred in **both skeptics' salvaged form**: the coloured-noise lever stays the ledger's named OPEN item, but its 6-line edit ships as *plumbing with no verdict attached*, and the experiment is replaced, because as proposed its kill criterion ("a variance-matched white arm matches the OU arm's D1 count") would have closed a live lever on a 0-vs-0 count of a banned metric, and its "must move" numbers (τ_int, the window-mean gain) are closed-form properties of AR(1) with ρ=exp(−1/36) — they test the RNG.

- [x] **D2a (free, decisive, do it first): measure `log_std.exp()` on the pi path.** — **DONE (§9), and it
  splits the pre-registration.** `controllers/rl/_diag_actor_std.py`. σ_ail = **0.67–0.90**, i.e. 4× the
  recorded 0.21 and 13–18× above `min_std` ⇒ **the actor-std-floor sub-lever is CLOSED** (it could never
  bind). But N_eff = **41.2 / 33.9** against a white-noise prediction of exactly **36**, and
  P(|1.5 s window mean| > 0.5) = **0.000 in all 40 channel-views** ⇒ the limiter is **temporal
  correlation, not amplitude**, so the coherence framing survives on a corrected basis and D2b is live.
  Two of my own in-flight hypotheses were refuted by the run's own phase-matched control — see §9
  findings 5 and 6 before quoting any `|mu|`/rail number from it.
- [x] **D2b (free): land the eps threading as a correctness fix, claim no verdict.** — **DONE (§9).**
  `world_model.py:83-97`, `agent.py` (`COLOURED_CHANNELS`, `_explore_eps`), `config.py:54-77`, 5 tests,
  `141 passed`. White stays bit-identical. **Measurement changed the recommended arm:** use `ou` τ=36
  (N_eff 1.41), **not** `pink` β=1 — β=1 is the weakest coloured option (2.20) so a null from it would be
  a weak null. Table in §9 and `config.py`.
  `world_model.py:87` `eps = torch.randn_like(mu) if eps is None else eps`; `agent.py:180-182` pass `self._explore_eps()` when `not eval_mode`, keeping the coloured component on channels **{0,1} only** and substituting a fresh white draw on {2,3,4} — OU/pink on a[4] latches the trigger (`fire` at agent-space ≥ 0.0) into a magazine dump that would confound with [[reward-v10-ammo-dump]], and on a[3] holds throttle through the bleed phase. Injection must be **pre-squash**: `pi()` returns an already-tanh'd `mu`. (No `world_model.py` edit is strictly required — tanh is bijective, so `tanh(atanh(mu) + log_std.exp()*eps)` in `act()` is equivalent; threading is just cleaner.) **D2a made pre-squash mandatory rather than merely tidy:** `dev_post ≈ σ·(1−mu²)`, verified to within 13% on a railed channel, so post-squash noise on a channel whose mean sits at the rail is a *measured* no-op — and roll is railed 0.45–0.73 of the opening 138 steps in **every** checkpoint and cell. **D2a also pins τ by measurement:** to move N_eff from the measured 36 to ≲2 needs **τ ≈ 0.5–0.75 s** (a ≈ 0.92–0.95 per 41.7 ms step); do not hand-pick it. And do **not** pair this with an actor-side std floor — D2a closed that sub-lever (σ is already 13–18× `min_std`). Two unit tests: `eval_mode=True` output bit-identical to today, and nonzero lag-12/36 autocorrelation of the **flown** action on the pi branch. Today the run header can record `explore_noise='pink'` while sampling white — that is the defect being fixed.
- [x] **D2c (free): the readout, on the executed action.** — **DONE (§9), and the lever CLEARS.**
  `_diag_actor_std.py --arm exec`. P(1.5 s window mean roll deviation > 0.5) went **0.000 → 0.256**
  (0.102 past 0.75), N_eff 8.2 → 1.2, closed loop. Pre-squash injection reaches the railed load axis too
  (`elev` P>0.25 = 0.325). ⚠ Do not read exec-arm N_eff on an *uncoloured* channel as a plumbing check —
  it is confounded by squash-gain drift; the pinned-`mu` unit test is the clean one.
  Every sanctioned probe hardcodes greedy (`_bfm_verdict_tmp.py:132`, `eval_policy.py:205`), so add a `--stochastic-arm` that logs the **flown** action or the lever is unmeasurable by construction. Re-run `_refute_noise_band.py` on the pi branch and report `P(|exec 36-step window mean| > D)` for D ∈ {0.25, 0.5, 0.75}, EXEC vs greedy vs noise — post-squash, with the tanh Jacobian `(1−a²)` accounted for, since greedy ail std is 0.63 and saturation attenuates the flown deviation hardest exactly where commands are large. This statistic has **no floor at zero**, so a null is distinguishable from a no-op.
  **Pre-registered in advance:** a null on D1/D2 event counts does **not** close the coloured-noise family (`handoff:894-898`).
  **cost:** free (CPU minutes to hours) for all three; ~a day for the plumbing and tests.

- [x] **D2d — ✅ CLOSED 2026-09-06, NEGATIVE. RESULT IN §9. Both arms ran; the treatment ties the incumbent
  (OU 11 kills vs c96 10 over 24 greedy eps, Fisher **p = 1.0000**) and `D1 = D2 = 0.00` on **all three**
  arms ⇒ the coloured-exploration seam D2b/D2c landed converts to ZERO displacement rolls and ZERO yo-yos in
  the greedy policy. The 11-vs-5 win over its own white control is **not significant (p = 0.1246)** and that
  control is triply confounded. ⇒ coloured exploration is LANDED but does not CONVERT; the exploration family
  is closed as a scoring lever.** Original box:
  <details><summary>as written before the run</summary>

  **D2d (one run): the verdict. Plumbing is VERIFIED launchable; the CELL is the open decision.**
  **Launchability checked 2026-09-06 and it is complete** — the [[outcome-reward-invisible-switch]] class
  does *not* bite here: `--explore-noise {white,pink,ou}` / `--explore-beta` / `--explore-tau-steps` exist
  in argparse (`train_parallel.py:4108-4123`), propagate through `build_config` (`train.py:131-133`) into
  `TDMPC2Config`, **and all three are in `_RESUME_OVERRIDABLE` (`:3947-3949`)**, so a warm `--resume`
  will print and apply the change instead of silently keeping the checkpoint's `white`. Collection-side
  construction is also live: `TDMPC2.__init__` builds `self.explore_noise` from cfg (`agent.py:165`) and
  `act()` resets it on `t0=True` (`:201`), so each episode gets a fresh correlated trajectory. The
  refuted σ=0.21/P≈1e-50 arithmetic that used to sit in the flag's own comment block has been replaced
  with the D2a/D2b/D2c measurements *in source*, so the next reader cannot re-derive it.
  **Arm:** `--explore-noise ou --explore-tau-steps 36` (N_eff 1.41). Not `pink --explore-beta 1`.
  **⚠ The open decision, and it must be settled before launch: WHICH LADDER.** The §6 amendment exempts
  D2 from the §10.5 ordering *only on the chase*, because that is the cell where the reward already
  prefers the max-g pull by +47 to +57/episode and the policy declines anyway. That exemption is about
  the **training signal**, not just the verdict geometry — a run that *trains* on `energy-merge` and is
  merely *judged* on the chase is still collecting against the merge's −2.94 charge, so a null from it
  would be uninterpretable and the exemption would not apply. Training on `chase-t3-killgate` is the
  form the amendment actually licenses. Against that: [[curriculum-93-chase-autopsy]] is the recorded
  control showing **5M more chase steps ERODES a warm c87**, so a warm chase run has a prior of decay,
  and c96's 45 kills are the thing at risk. Two admissible resolutions — pick one and say why:
  (i) `chase-t3-killgate` warm from c96, accepting the c93 erosion prior, with c96 itself as the
  no-change control (the coloured arm has to beat c93's *decay*, not c96's peak);
  (ii) c95's `energy-merge` line verbatim + `--rehearse-workers` already anchoring the chase finish,
  which keeps the base safe but re-imports the merge confound and therefore may only be read as
  "did it hurt", never as "the lever is null".
  **Judging:** the standard greedy verdict plus `bfm_detect` with the c96/c95/c87 D6 set in one
  invocation; `--outcome-reward` on the launch line; verify `collected ≈ 5M`.

  ### ▶ LAUNCHED 2026-09-06 — decision (i), with one improvement to the control

  **Ladder: (i) `chase-t3-killgate` warm from c96.** Why: the amendment's exemption is about the
  *training* ladder and only the chase gets it; the chase is the measured discovery-limited cell (the
  reward already pays +47 to +57 for the max-g pull and the policy declines it — [[reward-charge-structure]]);
  and horn (ii) is readable only as "did it hurt", which cannot settle a lever.

  **The improvement the box as written got wrong:** it nominated **c96 itself** as the no-change control.
  c96 is not a control for "5M more chase steps" — it is the *starting point*, so it cannot separate the
  coloured noise from the erosion. And c93 is not a substitute: different lineage (warm from **c87**) and
  it predates B1's honest WEZ, so it was trained and scored against a different reward function. ⇒ run
  **two matched 5M arms sequentially on the identical line**, single delta:

  - **ARM A (control):** `--explore-noise white` — passed **explicitly** so the header prints it
    (it logs `[resume] no-op (already matches the checkpoint)`, which is the point).
    `--save runs/curriculum-97-white/run.pt`
  - **ARM B (treatment):** `--explore-noise ou --explore-tau-steps 36`.
    `--save runs/curriculum-97-ou/run.pt`
  - **shared line** (matched to c93's proven chase launch): `--resume runs/curriculum-96/run.pt --ladder
    chase-t3-killgate --start-stage 0 --outcome-reward --workers 6 --rehearse-workers 2 --jitter 0.5
    --steps 5000000 --max-time-s 60 --stage-backstop-steps 12000000 --gate-window 60
    --gate-min-episodes 80 --probe-every 6 --device cuda --save-every 250000`

  **Three pre-launch checks, all resolved (do not re-derive):**
  1. **`--max-time-s 60` is required and does NOT move γ.** c96's checkpoint carries
     `episode_length 4320` (180 s) and `episode_length` **is** in `_RESUME_OVERRIDABLE` via
     `{--max-time-s, --rate-hz}` (`train_parallel.py:3942`), so the flag applies and the header prints
     `[resume] cfg overridden from CLI — episode_length: 4320 -> 1440`. The [[curriculum-88-discount-disproof]]
     trap does **not** fire: `cfg.discount is None`, so γ comes from `resolved_discount()`
     (`config.py:116-121`), and **both** 4320 and 1440 clamp to `discount_max = 0.995`. Header confirms
     `discount 0.9950`, unchanged. Sequential arms are matched anyway.
  2. **`seed_steps` stays 5000**, inherited from the c87→c93→c96 lineage in the checkpoint, so
     `resolved_seed_steps()` (which would give 5·1440 = 7200) is never reached. Identical in both arms and
     identical to c93's line. Note `seed_steps` is a *pretraining update count* here, not random actions.
  3. **The merge-only flags are dropped** (`--merge-lateral-m`, `--me-dalt-ft`, the whole
     `--merge-override-*` family) — they belong to `energy-merge`.

  **Arm A header verified:** `episode_length 1440`, `discount 0.9950`, `seed_steps 5000`,
  `reward OUTCOME`, `obs 41 act 5`, `start_stage 0 (RESUMED)`. Sequential on an idle RTX 5080,
  ~1 h/arm. Logs: `runs/curriculum-97-{white,ou}/train.log`.
  </details>

---

#### D3 — Winnability against the TRUE simulator (no GPU, no training, ordering-independent)

Preferred in **both skeptics' salvaged form** — four fixes, all load-bearing. The original planned 3 s (72 steps) against a terminal p_hit/lay cost, which is *the charge under test* and is identically flat beyond 1200 m, scored HP-dealt with no death term (whose recorded optimum is a suicide lead turn: 11 HP, dAGL −3227, decked 2/4), used a chase control that cannot fail for the reason the merge fails, and baselined against a chase-aspect-60 number on a merge cell whose true baseline is 0 HP.

This is the only box that can invalidate the program's framing rather than move a metric, and it touches neither the reward nor the learner, so **it is not subject to §10.5's ordering and can run in parallel with D1.**

- [x] **D3a (nearly free): the analytic dwell budget, first.** — **DONE 2026-09-06: PARTIAL (§9).** The
  measured half stands (gun constants verified against Rust; `gun.accum` is zeroed on a released trigger;
  **0.00 s achievable dwell and E[hits] = 0.00 in 16/16 arm × cell combos over 752.9 s of sub-600 m
  exposure**; learner α cone p99 12.2-12.7°; T3 break radius 869-1204 m). **The named band R\* = 271-353 m
  does NOT stand** — it appears in no A1 row of any artifact, `ALPHA_CEIL_DEG = 16.7` is hardcoded with no
  flag, and the only A1 table prints 403/362/365/499 m. Three headline sentences are falsified by
  seed-matched re-runs of D3a's own code, the one-belt bound is 700 m not 673 m, `E[hits]` has no magazine
  cap, `_coturn` treats n and α as independent, the mandated D6 control set was omitted, and **the file is
  not reproducible** (`hash(cell)` seeds under randomised `PYTHONHASHSEED`). ⇒ the honest verdict is **P2:
  "no dwell budget found under model M = the four scripted trackers"**, which bounds achievability from
  below only and **motivates no task change**.
  ~50 hits/kill at the measured ~82 rounds/s = **~0.6 s** of sub-8 m lay at 300 m (p_hit 0.98) rising to **~3.0 s** at 600 m (p=0.20), against a best-ever merge `tHit` of **0.10 s** — a 6-30× gap. Compute how many seconds of sub-600 m, sub-8 m lay a 795 m-radius F-16 can hold against the T3's measured break. This may settle winnability on paper for a fraction of the cost.
- [x] **D3b — ✅ CLOSED 2026-09-06, RESULT IN §9. The pre-registered NEGATIVE fired on both cells and is the
  only admissible statement about the out-of-plane half: "no solution found at horizon H under objective J."
  ⚠ BUT the box bundled two questions and the data separates them: the merge KILL is reachable (merge1500,
  **5 kills / 8 seeds**, pooled hp 99.0, `tLeth` 0.521 s, closing to **8 m**, against **0.0 HP from all three
  controls**) while the out-of-plane DEMONSTRATION is not (**0 loaded D1, 0 clean D2 in 15,187 rollouts**).
  merge800 found nothing at all (0 kills, `tLeth` 0.083 s). ⇒ this is the THIRD independent measurement that
  killing in this environment does not require BFM — and the strongest, because it is not a learner that
  failed to acquire BFM, it is an unconstrained optimiser that never needed it.** Original box:
  <details><summary>as written before the search</summary>

  **D3b: `controllers/rl/_diag_true_mpc.py`.**
  **change:** the Rust env exposes only `new/agent_ids/reset/step` (`crates/astrum-env/src/lib.rs:190-224`), so use prefix replay — `env.py:172-175` rebuilds `DogfightEnv` from a seeded TOML, and determinism is already pinned by `crates/astrum-sim/tests/env.rs:179-208` and independently measured this session at **max|div| = 0.000e+00 over 600 replayed control steps**. ~~Throughput measured at 40,847 ticks/s = 340× realtime (2.94 ms wall per sim-second, 10.9 ms warm build+reset)~~ ⚠ **CORRECTED 2026-09-06 — that figure is wrong by ~30×.** `_diag_true_mpc.py`'s own precheck measures **1,327–1,354 sim ticks/s = 11.1–11.3× realtime (3.69–3.77 ms per 5-tick control step)** on three independent runs, which agrees with the trainer's own `astrum-sim` banner (786–1,109 Hz achieved under load) and disagrees with nothing. Cost is still not an excuse for a short horizon, but ⚠ **my own restated cost was ALSO wrong and is corrected here from the running search's per-generation timings.** A 24 s / 576-step rollout is 2.1 s, so popsize 180 × 8 seeds is 3,024 s ÷ 6 workers ≈ **8.5 min for ONE generation at full seed evaluation** — "~57 min per cell" was one generation mislabelled as forty. The real cost is **load-dependent, because `seed_skip` evaluates all 8 seeds only for candidates that FIRE**: measured 81 s at gen 0 (1 firing) rising to 171 s at gen 6 (36 firing), so the bill grows as the search succeeds. Budget **~3–6 h per cell**, not one hour. Shard it; JSBSim is process-global, so one env per process. JSBSim is process-global: one env per process, shard over 6.
  The four required fixes: **(1) horizon above the maneuver band** — one open-loop parameterised trajectory over the whole 15-30 s, or ≥12 s receding; never 3 s, which is below §10.4's own 120-360-step band. **(2) terminal-outcome objective only** — cumulative `tLeth` (sub-8 m lay **and** rng ≤ `bfm_detect.LETHAL_RANGE_M` = 600) or kill, with `hard_deck` scoring −inf and altitude spent priced; **never** a per-step or terminal aim/p_hit potential (`range_factor` is `p_hit_range(rng)`, 0.002 at 900 m and 0.000 past 1200 m, so a 3000 m merge run-in has zero gradient and all samples tie through exactly the one-circle/two-circle commit window). **(3) budget to the dimension and marginalise the PRNG** — CMA-ES with population ≥ 10·dim (18 dims at 6 segments × 3 channels; 128 evaluations is 7 per dimension, i.e. local perturbation), and score every candidate as the **mean over ≥8 `sim.seed` values**, because gun dispersion is a shared seeded SplitMix64 advanced per round fired (`combat/resources.rs:44-56`, `systems.rs:29,75`) at 8 mrad = 4.8 m 1σ at 600 m against `HIT_RADIUS_M` 8.0 — a max over a stochastic objective across thousands of selections is the c13 lucky-noise gate. **(4) a control with power** — the chase cannot fail for the reason the merge can (c96 kills 6/6 there with D1=D2=D3=0.00, i.e. in-plane pursuit already solves it), so instead require the harness to rediscover the recorded hand-coded one-circle lead turn (aspect 14°, `alpha_req` 9°, 1 m lay, committed ~1000 m) and to beat an in-plane arm from a start where in-plane provably fails.
  **prediction, pre-registered as an existence / lower-bound search:** either it produces a trajectory with ≥1 kill from the neutral merge carrying a loaded D1 or D2 event and nonzero `tLeth` — in which case a feasible, physics-respecting out-of-plane demonstration exists and the search space has been learner-side all along — or it does not, and the *only* admissible statement is **"no solution found at horizon H under objective J."** It may **never** conclude "unwinnable" and may **never** motivate changing the task; a search bounds achievability from below only.
  **judging:** report the determinism precheck's max divergence; score the produced trajectory with the standard detector, holds priced in altitude; never HP dealt alone (the `damage` metric is closed as blind to HP taken, deaths and decks). Do not touch `R_EFF`, the scripted fire gates, or tier 3.
  **Do not claim this unblocks D-B.** `update_pi` (`agent.py:241-259`) has no BC/AWR/demo term and `buffer.py:42-47` has no demo label, so a demonstration provably trains only encoder/dynamics/reward/Q. The demo is necessary, not sufficient; the BC channel is a separate unbuilt change.
  **cost:** ~1 day to write, then ~1 h per cell per seed-set on 6 processes. No GPU, no training run, no benchmark change.
  </details>

---

#### D4 — The T3's DEFENSIVE release is gated on the learner's instantaneous pointing (free descriptive check on saved rollouts)

Preferred in **both skeptics' salvaged form**, which is a fraction of the original. The mechanism's novel observation is real and is absent from **all four** record locations (zero hits for FSM / DEFENSIVE / THREAT_CONE / `_transition` in this document; both prior T3 rewrites touched `_break`'s control law, never `_transition`). But two of its three channels are closed (see C-5 below) and the proposed fix, as written, produces a *permanently pinned break*, not a 3.5 s latch. What survives is a narrow, free question.

**DONE 2026-09-06 — FAIL: the nose-gate is a VERIFIED DEFECT but it is VACUOUS on the run cell (§9).**
The box's own FAIL branch fired: `P(ata > 65° | |χ| > 45 ∧ nz > 3)` = **0.0% (0/718)** on c96 asp55 — and
**26.5% / 34.6% / 66.6%** on the c95/c87 controls. Zero of c96's 6 releases land on a loaded out-of-plane
tick; all four flagged releases are at nz 1.9-2.4 with the lift vector past 90°; the in-window base rate
(15.1%) is **2.2× LOWER** than out-of-window (33.9%); the release-vs-`D5hit` anti-correlation is a pure
range artifact (release windows' credit-interval min range 688-722 m against a 600 m lethal bar); and per
1000 near-ticks c96 has the **lowest** release rate of the three checkpoints. ⇒ **Stop here as the box
instructed. Do not build `Tier3CommitOpponent`** — fixing the gate would make the benchmark easier while
explaining none of the ceiling. Side effect: the same range stratification **retracts B1's §9 point 3.**

- [x] **change:** on the **already-saved** greedy rollouts of the D6 control set (c96 / c95 / c87), log `Tier3Opponent.state` (as `_finish_trace_tmp.py:60,75-77` already does: `getattr(base.opponent, "state", "?")`) and `rg_t.ata` alongside the existing D1/D3 event windows. Condition **every** occupancy and energy statistic on `rng < THREAT_RANGE` (2500 m), because `THREAT_RANGE_EXIT = 3500` m will otherwise manufacture the trend — recorded NEUTRAL means are 3208 m (asp30) and 3421 m (asp45), both at the range gate, while only asp55's 2080 m is a genuine pointing-gate release.
- **prediction (falsifiable both ways):** during the **9 turning-room windows on the c96 asp55 wall cell** (the one non-degenerate D5 lift on record), does `rg_t.ata` ever exceed `THREAT_CONE_EXIT_RAD` = 65° (`opponent.py:385`), and does the T3 actually leave DEFENSIVE? χ (lift vector off the LOS plane) does **not** imply large ata — you can bank the lift vector far out of plane with the nose inside the cone. If ata stays under 65° through real loaded out-of-plane content, the mechanism is vacuous and dies for free.
- **judging:** if it *does* exceed it, record the nose-gate as a **verified defect** (currently in none of the four record locations) and stop there. Do not build a new opponent class in this box. If a fix is ever built: the cheap first form is a hysteresis widening of the existing exit (`THREAT_CONE_EXIT_RAD` 65° → ~100°) or a DEFENSIVE-only commit floor, in a `Tier3CommitOpponent(Tier3Opponent)` subclass wired as `opponent_tier=4` (`env.py:372-377` has the switch with an `else: Tier3Opponent` fallthrough; `train_parallel.py:1312` has the per-stage field; a new ladder row is needed since every chase/merge row is tier 0), tier 3 left byte-identical and the pinned opponent tests re-run. **`threat_clear` must become the positional complement** (e.g. `aspect > 110°` OR `rng > 3500`) or the two predicates stop being dead-banded and the T3 sits in unconditional max-g DEFENSIVE with an 8.3 ms NEUTRAL blip every commit interval — an easier bot, which is exactly what the candidate's own first negative control exists to catch. Any tier-4 **run** is scored against a self-graded benchmark and must therefore also cross-evaluate against the frozen tier 3, and is sequenced after D1b.
- **cost:** free (existing rollouts, ~20 lines).

---

#### D5 — Critic calibration: how far back does this critic price a kill it actually achieves? (free)

Preferred in **both skeptics' salvaged form**. The original gated itself on "the arm that eventually connects" from a `bfm.py` splice — A2 measured all five sticks at 0 kills / 0 rounds / 0 lethal ticks, so no such arm exists, and Q-along-a-scripted-arm is a category error (Q evaluates π, the MC return samples the stick, so a gap is guaranteed and means "off-policy", not "bootstrap too short"). The salvage measures the same quantity **on-policy**, where it is exact and genuinely new.

**DONE 2026-09-06 — INCONCLUSIVE (both verifiers), and it is the most productive box on the board (§9).**
The statistic is an **algebraic identity**: `gap_t ≡ −Σ_{k≥t} γ^(k−t)·td1_k`, so it can only report where
the one-step residuals sit, and they sit at the terminal — where `terminated = 1` means there is **no
bootstrap at all**. The wrong-way signature fired (82% of |G| at 1-2 s, `P(Q<G) = 1.00`, gap p50 −128.09 at
3-6 s on kills) but **C-6 stays closed on positive evidence**: the gap is maximal AT the terminal and
shrinks outward, decays as γ^n, the non-terminal residual is +0.002 p50 on 739/745 steps, and **both D6
controls show the identical γ^n structure with the sign flipped on a deck**. What fails is **terminal-reward
magnitude representation**, in two new source-level forms: **(1) decoder compression** — `two_hot_inv`
returns `symexp(E_p[bin])`, a log-mean, so with 0.56-0.86 of the Q mass above +50 and modal bin +147.41 the
shipped decode returns **median +35.4**; **(2) terminal-transition gradient starvation** — the terminal
stored index is reachable at `t=2` only, **6.82× less ρ-weight** than a mid-episode index. Both are now the
board's live candidates. Also: the pre-registered 3-6 s window **is** c96's episode opening (kills last
4.96-5.46 s), the docstring's "no policy gap by construction" is false (`_td_target` bootstraps with the
squashed **sample**, `world_model.py:96-98`), pessimism was reported per-backup and is −16.8 to −36.3 at
the fixed point, and `_estimate_value` rolls only 125 ms with corr **+0.1778** on the one kill cell.

- [x] **change:** `controllers/rl/_diag_credit_reach.py`. Roll c96 **greedy** (deterministic given the seed, so return-to-go is exact, not a sample) on two cells: aspect-30 chase (6/6 kills, 38 lethal ticks) and aspect-55 wall (0 kills, all timeouts). Per control step record `Q(z_t, a_t, return_type="avg")` — **never `"min"`**, which is `randperm(num_q)[:2].min()` (`world_model.py:105`), 2 of 5 heads per call, non-reproducible and deliberately pessimistic — against the exact discounted return-to-go at the γ **read out of the saved cfg**. Plot the gap against time-to-terminal (0.5 / 1 / 2 / 4 / 8 s), split by terminal type (kill / deck / timeout), horizons reported at 41.7 ms per control step.
  Second free measurement with code that already exists: call `_estimate_value` (`agent.py:248-268` — a γ-discounted H-step rollout of the learned dynamics + reward head with a terminal bootstrap, dead only behind the single `if self.cfg.mpc` guard at `agent.py:178`) against `Q(z, pi(z))` on the same greedy states.
- **prediction:** the critic is roughly calibrated — gap < 30% at 2 s, growing modestly out to 4-8 s — and `_estimate_value` agrees with `Q(z, pi(z))`, meaning the model+critic carry no additional multi-step information and n-step has nothing to add. **The wrong-way outcome that would matter:** a large, systematically *negative* gap at 3-6 s on trajectories that DO terminate in a kill, which would reopen bootstrap length as a live defect against `tdmpc2-credit-horizon-ceiling.md:65-67`.
- **judging:** report it as a calibration curve, explicitly **not** as BFM evidence; the interesting quantity is how the gap *scales* with time-to-terminal, not its level. Reward-term sums and discounted-return traces along a trajectory remain blacklisted (`bfm_detect.py:37-48`).
- **cost:** free, CPU minutes.

---

#### D6 — ρ alone, at H=3, one edit (last, weakest, and only if D1b moved the sign)

`rho` (`config.py:69`, 0.5) is the one algorithmic knob the ledger grades UNTESTED that also has a mechanism: `rho = pow(cfg.rho, arange(H))` caps trained latent-rollout depth at ~10 control steps for **every** H, which is why raising `--horizon` is closed and this is not. It is listed last and honestly weak: a skeptic showed `rho=1.0` is **not a single variable** — the same constant weights `update_pi` (`agent.py:375-378`, mean weight 0.469 → 1.000, i.e. a ~2.1× effective actor-LR increase, and LR perturbation of a warm peak is a recorded oscillator), the consistency loss (coef 20.0) and the value loss (0.1) simultaneously, and it hands the actor and critic latent depths 4-12 the dynamics model has never been trained to produce, at full weight, on the first update.

- [x] **D6a (free): the pre-check that can kill it.** — **DONE 2026-09-06: FAIL, and the leg is a
  DERIVATION, not the depth curve (§9).** At `horizon = 3`, `pow(cfg.rho, arange(H))` (`agent.py:317`) is
  evaluated at **k ∈ {0,1,2} only** and the consistency loop is `for t in range(H)` (`:325-327`), so the
  weight vector is literally **[1.0, 0.5, 0.25]** and depths 4-30 carry weight **exactly 0 at ρ=0.5 and at
  ρ=1.0 alike**. Nothing in the stack ever rolls the latent past **125 ms**. The box's own headline
  statistic is now **banned**: `ratio = predL2/persL2` is a **state-velocity** statistic driven by its
  denominator (identical model error 0.2053 vs 0.2121 against an 18.3× denominator swing; the denominator
  alternates with depth *parity* on c95), it is *decreasing* in depth on 6 of 9 cells so "first depth where
  ratio ≥ 1.0" is meaningless, and the three available normalisations give **three different orderings** for
  c96 vs c95. One thing survives on its own terms: latent error grows **sub-linearly** (`predL2 ∝ k^0.53-0.67`)
  with **no inflection at any depth** and cos 0.926-0.963 at depth 30 ⇒ **the dynamics model is not what
  prevents a multi-second commitment; the H=3 loop is**, which is a `horizon` question (closed on UTD cost),
  not a `rho` one.
- [x] ~~**D6b (one run, only after D1b has verifiably moved the sign):** `config.py:69` `rho: 0.5 → 1.0` at
  **H=3**, single edit.~~ — **CLOSED UNRUN 2026-09-06, on arithmetic, for free.** It buys **zero depth** and
  is not one variable: it is a ×1.714 rescale of the ρ-weighted total (effective `consistency_coef`
  **20.0 → 34.29**, with `reward_coef`/`value_coef` scaled alike), a **flattening** of the depth weights to
  [1,1,1], a ×1/1.714 **dilution** of the un-ρ-weighted termination share, and a **×2.133** rescale of the
  actor loss. Near-miss worth remembering: it would have shipped labelled "single edit". Three plumbing pieces are mandatory or it is a silent no-op: an argparse `--rho`, propagation of `args.rho` in `build_config` (overrides are diffed against `ckpt_cfg` at `train_parallel.py:4664-4668`), and an entry in `_RESUME_OVERRIDABLE` (`:3937-3954`) — plus a pinned assertion that a warm resume prints `rho: 0.5 -> 1.0` in the header, per the [[outcome-reward-invisible-switch]] class.
  **prediction:** trained rollout depth rises (log the effective-n histogram and the consistency-vs-value loss **shares**, since `consistency_coef` is unreachable warm), and greedy `tLeth` per episode rises with aspect-30 kills ≥ 5/6. **Wrong-way outcome:** greedy `tLeth` and rounds-fired-inside-600 m decline monotonically = drift into the passive basin.
  **judging:** pre-register a statistic that is **defined at base_hits = 0** — a Fisher exact on the 2×2 (event windows containing a lethal tick) × (full-length event-free windows containing one), which `bfm_detect` already returns as `n_event_windows / event_hits / n_base_windows / base_hits`. Never the D5 *ratio*: it reads `inf` at c96 asp55 (base rate exactly 0.000) and `nan` at asp30 (zero base windows, because the kills land under `D5_WINDOW_S`=8 s), and it becomes *less* defined as the intervention succeeds. **Tripwire, not a control:** the RunningScale/c88 tripwire will not fire here (n-step and ρ do not shock the target scale), so use a **passivity** tripwire on greedy `tLeth` and rounds-inside-600 m instead. **n-step is explicitly deferred**, and if ever run must be a *corrected* multi-step target (Retrace / tree-backup, or π-actions with model-generated intermediate rewards) — raw stored-action n-step over a uniform off-policy buffer with no importance weight anywhere in `buffer.py:142-176` shifts the fixed point from Q^π toward Q^μ, and the adverse direction is predictable: it devalues committed, loaded, low-energy, near-the-deck states, which is where a BFM commitment lives.
  **cost:** free pre-check; one 5M run (~40-60 min) at H=3 (no UTD cost).

---

#### Sequencing and the one contingency that matters

Run order: **D0 → D1a, D2a-c, D3a, D4, D5 in parallel (all free) → D1b (the first launch) → D3b (parallel, launch-independent) → D2's run and D6b (only after D1b).**

The §10.5 ordering is binding for D2 and D6: you cannot measure whether correlated exploration or deeper credit discovers BFM while the return for BFM is −2.94, and that confound invalidates a result **in either direction**. D3 and D4 are exempt — D3 never touches the reward or the learner, and D4 (in its surviving form) is a read of saved data.

⚠ **Amended 2026-09-06: that mandate is CELL-SPECIFIC, so D2 is *not* blocked on D1.** The −2.94 was measured on the **merge**; on the **chase**, `_cf_out.txt` shows the max-g pull arms scoring **+47 to +57 per episode better** than what the policy actually chose, and the policy declines it anyway (§10.2). So the chase is **not** reward-limited and the discovery levers are testable there **today, unconfounded, with no reward change**. This is what promoted D2a to first, and D2b/D6 inherit the same exemption *provided they are judged on the chase*. Progress: **D2a is done (§9)** — it closed the actor-std-floor sub-lever, kept the coloured-noise lever live on a corrected mechanism, and pinned τ.

**The contingency:** if **D1a fails** — if `inplane` or c87 clears the χ∧load∧¬in_zone gate as often as the out-of-plane arms — then stop-charging has no expressible time-local form on this instrument, §10.6's plan of record is empty, and the ordering mandate becomes vacuous rather than binding. In that case the board's centre of gravity moves to **D3** (is the task winnable at all, and what does the winning trajectory look like) and **D4** (is the environment itself pricing the maneuver negative), and the discovery levers D2/D6 stop being *confounded* and become merely *un-derisked* — runnable, but with no prior reason to expect them to convert. Write that outcome into §9 explicitly if it happens; it is the most informative single thing this board can produce.

---

#### ⚠ THE CONTINGENCY FIRED. Board state after the free tier (2026-09-06)

**D1a failed, on both clauses, at all nine threshold settings, upheld by two independent verifiers.** So,
by the pre-registration above and with no reinterpretation:

1. **Stop-charging has no expressible time-local form on this instrument.** §10.6's plan of record is
   **empty**. Do not propose another attitude-and-load gate — the family is closed, and c87's loaded
   descending break clears any such gate more often than the maneuver does, in both duty cycle *and* rate.
2. **The §10.5 ordering mandate is now VACUOUS, not binding.** There is no reward sign to fix first, so
   D2's run is unblocked **unconditionally** — not just on the chase. It was already exempt there; the
   ladder question in D2d is now the only thing standing between the board and a launch.
3. **D2/D6 stop being *confounded* and become merely *un-derisked*.** D6 then died anyway, on arithmetic
   (D6a/D6b above). **D2d was the last surviving pre-authored run on this board — it ran on 2026-09-06 and
   came back NEGATIVE (§9), so the board now has NO pre-authored run left.** Anything launched from here must
   be authored fresh, and E3 + D2d together point it at **representation**, not exploration and not reward.
4. **But the two boxes the contingency nominated as the new centre of gravity did not survive as
   nominated.** D4 came back **vacuous on the run cell** — the nose-gate is a real environment defect that
   the scoring checkpoint never triggers, so the environment is *not* pricing the maneuver negative through
   that channel. D3a came back **P2** — no dwell budget found *under a model of four broken-aim scripted
   trackers*, which the record's own 166 lethal ticks for c96 shows is a statement about the sticks, not
   the airframe. So the centre of gravity is **D3b**, which is now not optional: it is the only remaining
   instrument that can answer winnability, and it must first fix D3a's four defects (α ceiling as a swept
   flag, magazine cap, α-n coupling, `hash()`-free deterministic seeds).
5. **The board gained two candidates it did not author, both from D5, both source-level, both cheap.**
   Decoder compression (`two_hot_inv` returns a log-mean; the critic's modal Q bin is +147.41 and the
   shipped decode returns median +35.4) and terminal-transition gradient starvation (the kill transition
   carries **6.82× less** ρ-weight than a mid-episode one). Neither is a reward constant and neither is an
   exploration knob — which makes them the first structurally new levers since B1. **They belong on the
   next board, ahead of anything reward-shaped.**

**Net: 7 boxes, 5 closed this pass (D0 done, D1a/D1b closed, D4 closed, D5 inconclusive-but-productive,
D6a/D6b closed), 2 live — D2d (a run, ladder decision pending) and D3b (~1 day to write, no GPU).**
⛔ **SUPERSEDED 2026-09-06: ALL 12 BOXES ARE CLOSED — Phase D 7/7 (D2d NEGATIVE, D3b's negative half stands),
Phase E 5/5 (E1a FAIL, E1b closed, E2a CONFIRMED, E2b closed by measurement, E3 CLOSED with the outcome-blind
reward head). No live box, no pre-authored run.** The
board did exactly what §6's own honesty note predicted it would: it closed branches decisively for near-zero
GPU cost rather than moving a kill count. Five levers died for the price of ~83 minutes of CPU.

---

### Phase E — the two candidates D5 handed back (authored 2026-09-06, NOT yet skeptic-reviewed)

Both come out of D5's audits, neither was proposed by anyone, both are **source-level defects in how the
critic represents a kill** rather than reward constants or exploration knobs. That makes them the first
structurally new levers since B1 (`OBS_DIM 31→41` and the honest WEZ are the only two things that have ever
worked here, and both were **representation**, not reward). ⚠ Neither has been through the two-skeptic pass
every Phase-D box faced — do that before launching either.

#### E1 — the critic's decode throws away the kill (free pre-check, then one edit)

- [x] **E1a — DONE 2026-09-06. FAIL: the decoder-compression mechanism DOES NOT EXIST at the reported
  size. `_diag_e1a_decode.py`, three checkpoints × two cells in one invocation, CPU. Three of the four
  pre-registered kill conditions fired. E1b is CLOSED. Full record in §9.**
  Headline: `Δ = q_true − q_ship` has p50 **0.14** on the one kill-bearing cell (q_ship 41.12 → q_true
  41.29), max 1.66, and **0.3–0.5% of |Q| on every one of the six cells**. D5's "median +35.4 vs a modal
  bin of +147.41" is not reproducible: the measured modal bin is p50 **43.70** / p90 53.60, and
  `N_eff bins = exp(entropy)` is **1.25–2.63** (p50 1.83) — the predicted distribution *is* essentially a
  genuine two-hot, which is exactly the regime where M0 proves the codec **exact** (worst relative error
  1.4e-6). There is no spread for a log-mean to compress.
  Original box text, kept for the record: *"D5 measured mass 0.56-0.86 above +50 with modal bin +147.41
  and a shipped decode of median +35.4; if the two estimators differ by ~4× on kill states and are ~equal
  on timeout states, the bias is real."* They do not differ by 4×; they differ by 0.4%.
- [x] ~~**E1b (one edit, only if E1a passes).**~~ **CLOSED 2026-09-06 by E1a's pre-registered gate** —
  K2 (ordering preserved: Spearman **0.99998–1.00000**, discordant pairs 0.006–0.130%, top-decile rank
  shift ≤ 0.07%) and K3 (the gap is the *same size* on spike-free cells) both fired. All three forms are
  off the table, including the "few lines" form (i). The bias is inherent to `symexp(E[symlog])`
  (`functional.py:53-67`); it is not a bug to patch but a scale choice to change. The cheap forms, in
  increasing cost: (i) decode as `Σ p_i · symexp(bin_i)` — same network, same targets, correct conditional
  expectation, a few lines; (ii) widen/re-space the bins so R_KILL = 150 is not on the tail; (iii) scale
  R_KILL down and the dense terms with it. ⚠ **(iii) is an objective shock under a converged policy** — the
  c88/c62 class — so it must be a from-cold or heavily-annealed change, and (i) is the honest first test
  because it changes the *readout* without moving the target scale.
  **Judging:** greedy verdict with the D6 set; `tLeth` (sub-8 m lay **∧** rng ≤ 600 m) and aspect-30 kills
  ≥ 5/6 as the passivity tripwire. **Do not** read the train-log `ret` — it is PBRS-dominated.

#### E2 — the transition that carries the kill is sampled ~7× too rarely (free measurement, then one edit)

- [x] **E2a — DONE 2026-09-06. CONFIRMED, with a corrected constant and a sharper mechanism than D5's.
  `_diag_e2a_buffer.py`, CPU-only (deliberately, so it did not contend with D2d for the GPU). **Zero of the
  six pre-registered kill conditions fired.** Full record in §9.**
  The ratio is **exactly 7.000×** (`1.75 / 0.25`), not 6.82× — 6.82 was a Monte-Carlo estimate of a
  quantity that is closed-form. It is **length-invariant**: 7.000 at every episode length from 20 to 1440.
  The three things E2a establishes that D5's number did not say:
  1. **The terminal is pinned at maximum depth.** 0 of 904 enumerated hits and 0 of **4,648** occurrences
     read off the shipped `buffer.sample()` tensor land at `t = 0` or `t = 1`. So the kill reward is
     supervised *only* at `zs[2]` — a latent two dynamics steps from the encoder — and never at `zs[0]`.
  2. **The negative control separates "depth" from "boundary".** The episode's FIRST transition is *also*
     reachable from exactly one start, but at `t = 0` with weight 1.0 ⇒ ratio only **1.75×**. So this is
     not "boundary indices are down-weighted"; it is specifically that the *episode-final* index is pinned
     at the most-discounted ρ position. Full profile: pos 1 = 1.00, pos 2 = 1.50, pos 3…T-2 = 1.75,
     T-1 = 0.75, **T = 0.25**.
  3. **The sampler itself is uniform** — `_sample_starts` draws every valid start equally (all |z| < 1.7
     over 49.8 M draws, 0 invalid starts ever drawn), so the profile is pure slice arithmetic, not a
     sampling bias. Which also means it is fixable in the sampler.
  ⚠ **Report the base rate with the ratio.** The terminal's share of the reward/value loss goes from
  **0.013% → 0.091%** (K4 did *not* fire, so "starved" is fair — but it is a 7× on 1/8000 of the loss, and
  §9's rule about `D5`-style ratios applies to this one too). Diet: **0.232 terminal transitions per
  256-slice batch = one kill every 4.3 updates**, at weight 0.25 and depth 2.
  ⚠ **D6a partly blunts the depth argument**: the dynamics model is accurate (cos 0.926–0.963 at depth 30),
  so `zs[2] ≈ encode(obs[2])` and "never supervised at depth 0" is milder than it sounds.
  **New finding E2a turned up that D5 did not claim:** `termination_loss` (`agent.py:337-341`) is a plain
  BCE mean and is **not** ρ-weighted, so for the termination head the defect is purely the reachability
  count (1 vs H): its positive-class rate is **3.03e-4, one positive per 3,305 elements**, and E2b would
  ×3 it. That is a candidate common cause for the recorded **dead termination head** (§10.4 /
  `tdmpc2-credit-horizon-ceiling`), which no prior box had a mechanism for.
- [x] **E2b — CLOSED (measured, not run), 2026-09-06.** The gate answered it for ~6 min of CPU instead of a
  5M run: **K2 failed on all three checkpoints by 5–19 SE (AUC 0.94–0.99)** ⇒ the head is not signal-starved
  ⇒ **half (b) is dead**, and the pre-check's `cfg.mpc = False` caps half (a) at the 0.013% → 0.091% share of
  the reward/value loss. **Do not spend a run on this box.** Half (a) survives as a **free rider**: form (i)
  (oversample terminal-hosting starts) is sampler-only, so if any future work touches `_sample_starts` for
  another reason, land it then under the c88 tripwire — it is not worth its own run but it is worth not
  losing. ⚠ **The gate's own numbers were later shown to be measured at an untrained query point (M7/M8,
  §9) — which REFUTES "the termination head is dead" and makes turning MPC on predicted-harmful, but does
  not revive half (b):** K2 is the gate that mattered and the head discriminates *better* at its trained
  depth. Full reasoning and the two overturned claims: §9 "M7/M8 RESOLVED".
  <details><summary>Original box text (the three ranked forms, kept for the record)</summary>

  ⚠ **E2a's depth histogram kills the cheapest option as
  written:** the terminal can only reach `t = 0` from start `s = T-1`, whose window spans slots
  `T-1, T, T+1, T+2` — i.e. it **straddles the episode boundary**, so positions after the terminal would
  train on the *next* episode's transitions. "Allow the terminal at `t=0`" is therefore not a one-line
  relaxation of the rejection rule; it needs a post-terminal validity **mask** threaded from `buffer` into
  `agent.update` (and into the ρ-weighted sums). Remaining options, cheapest first:
  (i) **oversample terminal-hosting starts** (`_sample_starts` mixes in starts drawn from the terminal set
  with probability p) — sampler-only, no loss change, fixes the *diet* and the termination-head count but
  not the depth or the ρ-weight; (ii) **up-weight terminal-bearing slices** by the measured 7× — one line
  in the loss, fixes the weight but not the depth; (iii) **straddle + mask** — fixes depth, weight and
  count, but touches `sample()`'s contract and `agent.update`'s three ρ-weighted sums.
  ⚠ All three change the effective data distribution or loss scale under a converged policy, so the **c88
  tripwire** applies — diff the target scale in the header and abort on a RunningScale jump. Note E2a
  retired the E1 interaction this box used to cite: there is no decode bias to propagate.
  ⚠ **DOWNGRADED by its own pre-check (2026-09-06, §9) — this is now a SMALL-SHARE fix and must be written
  up as one.** `cfg.mpc = False` in every run, so **nothing consumes the head's prediction**; its only
  route to behaviour is the auxiliary BCE gradient into the shared latent. E2b's case is therefore
  (a) the terminal's 0.013% → 0.091% share of the reward/value supervision plus (b) a ×3 auxiliary signal.
  Form (i) is the only one whose benefit survives the pre-check intact.
  **GATE BEFORE SPENDING THE RUN — now a committed harness, `_diag_e2b_term.py` (running 2026-09-06):**
  does the head discriminate real terminals at all (AUC), does it ever cross the hard latch, is it
  calibrated, and does its peak belief actually land ON the terminal tick (M5)? Pre-registered K1–K4 in the
  docstring; **K2 (AUC ≤ 0.60) is the only outcome that justifies half (b)** — if the head already
  discriminates, the auxiliary signal is not starved and E2b collapses to half (a) alone. D6 controls
  (c95, c87) run in the same invocation, because a head-quality number that also appears in c87 is a
  property of the architecture and the base rate, not of anything this program changed.
  </details>
- [x] **E3 — ✅ CLOSED 2026-09-06, RESULT IN §9. `K1 FAIL / K2 FAIL / K3 PASS / K4 FAIL on all three
  checkpoints` ⇒ the learned reward function is OUTCOME-BLIND, and c87 shows it too ⇒ architectural.
  Headline: the predicted kill-minus-death gap is 8.1 on c87 where the truth is 352.9 (43× compression).
  The MPC branch resolved to the BAD half — dying is attractive to a plan in 9/9 real deaths at +56..+83 ⇒
  "expect a null" is RETIRED and MPC must not be run until the terminal is priced. My own pre-registered
  density prediction was REFUTED (c87 has 7 deaths and still prices death at 0.1%); the corrected reading is
  SIGN-specific, not frequency-specific.** Original box, kept for the pre-registration record:
  <details><summary>as written before the run</summary>

  **Does the world model's REWARD head predict the terminal rewards at all?** E2b proved the *termination* head is healthy at its trained
  depth. The same question has never been asked of the **reward** head, and it is the bigger one: E2a measured
  the terminal transition at **0.013% of the reward/value loss, one per 4.3 updates**, so the head may simply
  never have learned that a kill pays `R_KILL = +150` or a deck costs `R_DECK = −200`.
  **Measure** `two_hot_inv(reward(z, a))` on real terminal transitions, at the **trained query point** (M7's
  lesson — `_zs = zs[:-1]` includes depth 0 for this head, unlike termination, so depth 0 is *legitimate*
  here; verify that from the slice index rather than assuming, and report a depth ramp anyway since the
  terminal is only *reachable* at `t = 2`).
  **Pre-register, and quote base rates not ratios:** (K1) predicted kill reward within 25% of +150;
  (K2) predicted deck reward within 25% of −200; (K3) a depth ramp, in case this head has the same query-point
  sensitivity; (K4) the dense-reward control — predicted vs actual on ordinary mid-fight ticks, which
  establishes the head is accurate *somewhere* and so isolates the terminal as the failure.
  **Why it matters, in priority order.** ⚠ **First, the scope, checked before the run and narrower than my
  first draft of this box claimed:** `_td_target` is called on the tuple straight out of `buffer.sample()`
  (`agent.py:308,:313`), so **the critic's TD target uses the GROUND-TRUTH reward** — the value path is *not*
  blind, and "every path is optimising a blind reward" is wrong. An outcome-blind reward head costs exactly
  two things: (1) **MPPI is unusable** (moot at `cfg.mpc = False`, decisive if MPC is switched on — see the
  branch below); (2) **one of the three gradients shaping the shared latent carries no outcome information** —
  the encoder is shaped by reward-prediction, Q and consistency, and `act()` is `pi(encode(obs))`, so a
  reward head blind to the outcome means the latent is organised around the outcome only via Q and
  consistency. That is a **representation** claim, which is the only family that has ever moved this program
  (c87 obs, c96 WEZ) — which makes it worth measuring, but it must be stated as that and not as more.
  (3) it decides the MPC branch (§9, M7/M8 item 3): a correct −200 ⇒ no planner defect and
  "expect a null" stands; a ≈0 ⇒ death is worth ≈ +66.5 to the elite set and MPC must not be run until
  `agent.py` prices the terminal. ⚠ **D6 controls (c95, c87) in the same invocation** — if c87 shows it too it
  is architectural, not something this program caused. ⚠ **This is a measurement, not a lever**: a failure
  here does **not** authorise a reward-constant change (γ and the reward constants are a coupled pair) — it
  authorises fixing the *supervision*, which is E2b form (i), already costed and sampler-only.
  **⏱ PRE-REGISTERED PREDICTION, written 2026-09-06 06:33 after the c96 block printed and BEFORE the c87
  control did** (recorded in advance so this is a test, not a post-hoc story). c96 prices the kill at **73%**
  of true (+111.77 vs +152.16, n = 27) but death at **2.7%** (−5.33 vs −200.10, n = 1) — in a sample where it
  killed 27 times and died once. **If the driver is DATA DENSITY** (the head prices whichever terminal its
  buffer actually contains), then c87 — which decks 29/29 in the re-baseline and kills twice — must show the
  **mirror image: DEATH priced well, KILL priced badly.** If c87 instead also prices kills well and death
  badly, the asymmetry is about the bonus's *sign or magnitude*, not its frequency, and the density story is
  dead. The depth-2 specificity already stands independently of which way this goes: c96's kill ramp is
  +1.13 / +2.07 / **+111.77** / +35.95 at depths 0/1/2/3 — a peak exactly AT the trained depth and worse on
  **both** sides, so it cannot be monotone `next()` drift.
  </details>

**Why these two are worth more than they look.** §10.4's diagnosis was "nothing in the stack can represent
a multi-second commitment", and D6a just showed the dynamics model is **not** the limiter (sub-linear error
growth, no inflection out to 30 steps, cos 0.926-0.963 at 1.25 s). What is left on the representation side
is the **value** path — and D5 found two independent defects in exactly that path, both of which
specifically attenuate the terminal kill reward, which is the only signal that could ever pay for a
maneuver that costs geometry now to win later.

⚠ **AMENDED 2026-09-06, same day, after E1a ran: one of the two is gone.** E1 was **not** a defect — the
decode gap is 0.3–0.5% of |Q| and preserves the state ordering to ρ = 0.99999, so it re-prices nothing.
**E2 is untouched by that result** (it is an arithmetic property of `buffer._sample_starts`, independent of
the decode) and is now the only live Phase-E box. The paragraph above should be read as: *the dynamics model
is not the limiter, so the value path is where to look — and one of the two things D5 found there does not
exist.* Both of D5's mechanisms came from **agent-reported numbers with no committed harness behind them**;
E1a is the first committed measurement of either, and it contradicted the one it tested. **Run E2a before
believing E2's 6.82× either.**

⚠ **AMENDED AGAIN 2026-09-06, after E2a ran.** E2a **confirmed** its half — so the two D5 claims split 1–1,
which is the right reason to have measured both rather than binning both on E1a's failure. The corrected
statement of the Phase-E premise: *the dynamics model is not the limiter (D6a); of the two value-path defects
D5 reported, the decode one does not exist (E1a) and the replay one is real but is exactly 7.000×, moves the
terminal from 0.013% to 0.091% of the loss, and its most valuable consequence is probably the un-ρ-weighted
**termination head** seeing one positive per 3,305 elements.* E2b is the only live Phase-E box.

---

#### Checked and closed in this pass — do not re-propose

**C-1. Freeze-and-flush: hold the four geometry potentials across a commitment and pay the netted delta on exit.** Closed on the fix-independent magnitude leg of §3 row 5 (`handoff:64` "~5% delay tax on something worth 20× more") and by direct measurement: implemented verbatim (`controllers/rl/_refute_freezeflush.py`) on a real kinematic rollout, total discounted relief is **+0.106 / +0.582 / +2.783** at 2/4/6 s against **+92.71** for a kill — versus the claimed +14.378, because real out-of-plane rolls are *not* recovered excursions (the roll ends at 952 m vs a 551 m no-excursion baseline; the potentials are functions of relative geometry, not attitude). It also breaches its own stated kill criterion by 4.5-9× (max single-step emission 22.6 / 45.3 vs its own 5.0 line) and its sum-preservation safety argument fails out of zone by ~25 on `K_LEAD` — at which point it *is* the closed PATCH-B de-ratcheting (`handoff:65`, six-disproofs item 6). Where the relief would be large the mechanism also inverts: γ^(24T) discounts an *unrecovered* excursion identically, so at c95's measured 55.6 s loaded hold the bill for discarding all geometry arrives at 0.13% of shipped.

**C-2. `--horizon 12` + `rho=1.0` + n-step as a joint move.** Closed by §10.5's ordering (`handoff:1032-1035`) plus §3 row 2 (H closed for depth, and it costs a measured 2.1-4.2× UTD). Independently: the ρ leg is **inverted** — under `n_t = min(H, H−t)` the deepest return sits at t=0 where ρ=0.5 already gives maximum weight, so weighted-mean n is 11.0 at ρ=0.5 vs **6.5** at ρ=1.0; the three arms differ by 6.5× in effective actor LR on top of return depth; and the headline metric (D5 ratio ≥ 1.3) has no measured ~1.0 baseline anywhere (recorded values are `inf` and, at n=1, 10.78 for a checkpoint that decks 5/5). ρ survives **only** as the de-scoped single edit in D6.

**C-3. Restoring policy-invariance to the two `engaged` energy gates (`reward.py:316`, `:371`).** Closed twice. The premise is inverted in the strong direction: `_diag_counterfactual_pull_tmp.py` / `_cf_out.txt:50130-50132` measured `energy` **+6.8/+7.2/+7.5** and `rel_energy` **+1.2/+1.9/+2.5** per episode across BASE/PURE/LAG arms — the two energy terms are the only *positive* dense terms and they get more positive the harder and more out-of-plane you fly (`curriculum-93-chase-autopsy.md:60-64`: "refuted in the strong direction"). And the gate is inert where it was claimed to bite: measured on greedy c96 at the exact training geometry, deleted gain is median **0.63** / mean 3.07 undiscounted = **+0.358 discounted against |J| ≈ 560, i.e. 0.06% of the return**, and **exactly 0.000 on the entire chase benchmark** (0 ticks outside the gate at aspect 30 and 55). The proposed repair also multiplies the loss side by `w(rng)`, softening the "always charge energy losses" anti-deck property (`reward.py:311-313`) on a cell that decks 4/5, and re-pays the Es gain outside `TRACK_RANGE` that reward-v25/c75 shipped a fix to *stop* ("left the bubble climbing away with endRelEs 4000+"). **The whole "restore invariance" branch of stop-charging is closed as a family.** Its one true side-finding is banked in D0.

**C-4. Freezing the policy's roll-rate command (`a[0]`) for 1-2 s at the collector.** Closed by the D4 correction (`handoff:620-624`, `bfm_detect.py:505-522`): `a[0]` is a normalized roll-*rate* command, so a displacement roll is a pulse-then-null and holding |a[0]|>0.5 for 1.5 s commands 55-170° of continuous roll — a jet spinning about its velocity vector, i.e. *departing* the maneuver; the surviving statistic is `longest_rate_cmd_run_s` = **dither, where low is correct**, and the proposal maximizes it. Independently: at `hold_p=0.06/step` with k~U[24,48] the realized duty cycle is **68%** of every non-probe control step on a frozen two-axis command (81% at 0.12) — a majority-of-collection open-loop takeover of both primary flight axes, nearer c62's whole-episode override (THE recorded destabilizer) than the fire-override precedent invoked. Also: the premise is contradicted by data in hand (the policy already emits `elev` τ_int 4.8 s, |elev|>0.5 held 29.7 s, loaded chi holds to 5.40 s — all with D1=D2=0.00), and the acceptance gate (≥0.5 loaded D1/ep) exceeds the measured ceiling of the best deliberate demonstration ever flown (A2: 0.00 for every arm at every threshold). **The pulse-then-null variant survives, as an arm of D2c only** — freeze `a[1]` (the Nz/g command, which is state-like and holds the load) and *pulse* `a[0]` then command ~0, with a per-episode onset budget and the realized duty cycle printed in the header. Two source defects to fix if it is ever implemented: five sites overwrite `action[:4]` **after** the proposed insertion point (`train_parallel.py:980, :1037, :1102, :1110, :1119`), and `_base` is None-initialised at `:797`/`:815` with every existing consumer guarding it (`:962`, `:1003`).

**C-5. Two of the three channels of the T3-release mechanism.** The "release gifts the bandit energy" half is closed: the **gated** break keeps speed (`opponent.py:440`, V 158 → 203 m/s), and post-fix FSM occupancy on the same c87 checkpoint reads DEFENSIVE 46-60% at V 216-224 vs NEUTRAL 184-213 with `ENERGY_RECOVERY` absent (`_who_fixed.txt:6-24` vs the pre-fix `_who_out.txt:6-16`) — so `_pursue`'s `V_PURSUIT=200` cap makes the release a *deceleration*, and the "`_energy_recover` = sitting duck" payoff existed only under the **voided** ungated break. The "release gifts the bandit a shot" half is closed outright: `wez_fire` is state-independent (`opponent.py:305`, called unconditionally at `:493-494`) — the T3 fires while breaking. The surviving residual ("the release cancels turning room being built") is D4.

**C-6. Bootstrap length as a novel mechanism.** Closed by `tdmpc2-credit-horizon-ceiling.md:65-67`, which already states it by name with the same arithmetic: "~25 Bellman backups per transition = credit propagates ~25 control steps ~1 s of flight, with strictly 1-step TD (**no n-step, no lambda, no prioritisation**)". The absence is recorded; the *fix* is closed by §10.5 ordering plus the measured return ordering (−4.39 discounted for a 4 s displacement roll against +92.71 for a kill: at γ=0.995 a roll-then-kill at 4-6 s is already 20× positive, so credit reach is not what suppresses it). The MC half of the proposed diagnostic is also already built (`_refute_diff.py:9-14, 97, 109`, discounted MC at γ=0.995 over four arms). The on-policy calibration curve survives as D5.

**C-7. Two judging patterns, closed as a class.** `D_straight` (`_refute_diff.py:139-140`, nz≈1) as the negative control for any load-gated waiver: excluded arithmetically, cannot fail, proves nothing — the discriminating arm is a **loaded nose-off descending break** (replay c87). And D1/D2/D3 per-episode **counts** as any lever's success criterion: already banned (`handoff:894-898`; c95 at asp55 has more events than c96 and zero lethal ticks in the whole cell), and additionally underpowered at the episode counts this program actually runs (4-8/arm), where a 0.1/ep effect is a coin flip. Any future proposal whose pass bar is a count, or whose kill criterion is a 0-vs-0 comparison, is not an experiment.

---

### Phase F — authored by the user 2026-09-06 (the board's only live entry)

- [x] **F1 — CLOSED NEGATIVE 2026-09-06 by the replay autopsy: the artifact cannot be an opponent.**
  ±83 ms of its own clock, or 10% of start geometry, takes it to **0/8 kills and 0.0 HP**. Original
  authoring below.

  **F1. Keep the D3b CMA-ES solution as an "expert" opponent, since it dominates in this sim.**
  User decision, 2026-09-06, in response to D3b's positive half (5 kills / 8 seeds at merge1500 where
  the hand-coded one-circle, two-circle and in-plane arms all dealt **0.0 HP**). Banked as a direction,
  **not yet scoped into a run.** Four structural facts must be designed around, not discovered later:

  1. **It is open-loop and opponent-blind by construction.** `Genome.act` (`_diag_true_mpc.py:199-224`)
     reads `me.phi` and nothing else; the inner P-controller is own-attitude only. So as an *opponent*
     it replays one fixed 24 s sequence every episode no matter what the learner does. It is a
     **scripted trajectory**, not a policy, and cannot be called an expert in the general sense.
  2. **It is tuned to exactly one start.** merge1500 = separation 3000 m, aspect 30°, `opp_heading` 180°,
     `me_dalt_ft` +2000, `alt_ft` 15000, and the 8 search seeds. `merge800` under the *same* search
     found **nothing** (0 kills, best fitness never above +3.8 in 40 gens) ⇒ the solution is
     geometry-specific and there is no reason to expect transfer off its own cell.
  3. **A fixed opponent is memorisable, which is the failure mode this program already has.** It is
     the T3 disease with the reactivity removed. Any use must pair it with either randomised starts,
     a library of per-geometry solutions, or a mix against the reacting T3 — never as the sole bandit.
  4. **The role swap is not free.** The genome flew the *learner's* airframe from the learner's side
     **with** the +2000 ft perch. Mounting it on the opponent hands the bandit the energy edge unless
     the cell is mirrored. Decide deliberately which side keeps the perch.

  **What it is genuinely good for, and the reason to keep it:** the current T3 deals ~nothing from the
  merge, so the learner has never faced a merge that can actually kill it. A bandit flying this
  trajectory *can*. That makes it a valuable **sparring partner / threat generator**, which is a
  different and more defensible claim than "expert demonstrator" — and it sidesteps the
  `d3b_best.npz`-consumes-nothing problem entirely, because a threat opponent needs no BC channel
  (`update_pi` has no BC/AWR term; `buffer.py:42-47` has no demo label).

  ⛔ **THE AUTOPSY CAME BACK AND F1 IS UNSUPPORTABLE AS WRITTEN (§9, 2026-09-06).** The trajectory does
  not survive **±83 ms of its own segment clock** (5/8 kills, 99.0 HP → **0/8, 0.0 HP**) and gives
  **0/8 kills and exactly 0.0 HP on 10 of 10 perturbed start geometries**; across 34 perturbed arms × 8
  seeds, 33 arms produce zero kills and 31 produce exactly 0.0 HP. Kill-vs-no-kill on the unperturbed
  arm is decided by **gun dispersion**, and the fire discipline is a **hand-coded oracle** (`_trigger`)
  that is not among the 18 optimised numbers. ⇒ **"it dominates in this sim" is false as stated: it
  dominates one start, to within 83 ms.** It is one alignment, not a policy, and an opponent must be
  robust to where the fight starts. **Do not build an opponent on `d3b_best.npz`.** The only salvageable
  form is *re-run the search per geometry to build a library*, which is a new program (cost: ~6,080 s
  per cell at dim 18) and not a reuse of this artifact — and it inherits every caveat above.

- [x] **F2 — the detector positive control + the perch re-fly. DONE, and it broke the instrument open.**
  D1 = **8/8** on four merge arms each dealing **0.0 HP** and **0/8** on the only arm that kills ⇒ on the
  merge, D1 was measuring the **pass**. F2b confirmed the `me_dalt_ft = 0` prediction (perch 0 ⇒ 0/8 kills,
  **0 rounds fired**, identical on all 8 seeds). See §9.
- [x] **F3 — the `(horizon, ρ)` pre-check. CLOSED BY ARITHMETIC in ~20 s, no GPU.** The terminal is pinned
  to reward index H−1 in exactly one window per episode ⇒ the gradient share caps at **7.00×** against the
  **43×** E3 needs, and the authored `H 3→9 / ρ 0.5→1.0` scores **0.78× — WORSE.** Replacement specified:
  a terminal importance weight `(1+λ·terminated)` on `reward_ce`, **λ=100**. See §9.
- [x] **F4 — the T3's turn plane.** |χ_T3| **1.7–4.6°**, 77–94% of its turn horizontal at 83–95° bank, and
  it **CLIMBS** ⇒ **the +2000 ft perch is given away by t ≈ 11 s.** The dive is **ours, and a CHOICE**:
  `bfm.LeadTurn` turns 13.7–14.1°/s **while gaining 339 m**. The lever is **bank angle**, not g. See §9.
- [x] **F5 — the vertical, over whole 180 s episodes.** The dive is real on net `dalt`, the **sign contrast
  is REFUTED** (the T3 descends too), and **"the reward pays for diving" is REFUTED** — all six vertical
  terms are negative and worth only **−25..−40** against returns of −364..−535 ⇒ **barely priced.**
  `K_BAND`'s episode sum is **−0.00 by γ=1 telescoping ⇒ a `K_BAND` edit is inert BY CONSTRUCTION.**
  Mechanism: a **zoom-stall.** See §9.
- [x] **F6 — is the yo-yo entry gate ever open on the merge?** ⚠ **My pre-registered P1 is HALF REFUTED:**
  the HIGH gate **does** open on a lag-pursuit ingress (joint 0.053, best tick aspect 83° / rng 897 m /
  +200 m/s). On `leadturn` it is shut on **RANGE** (blame 0.93), not the clause I predicted. **P3 confirmed
  — the LOW gate is open 39.4%** ⇒ the doctrine here is *cut the corner*, not *kill closure*. And
  **entering the yo-yo when its own gate opens throws away 123° of aspect** and still fires 0 rounds.
  ⚠ all six arms deck or dive ≥3.4 km; seeds bit-identical ⇒ **n = 1**. See §9.
- [x] **F7 — does the D1 repair repair D1?** Steps 1–4 landed in `bfm_detect.py` behind switches; the four
  0.0-HP merge arms go **8/8 → 0/8**, the ten chase arms stay 0, the killer arm stays 0 in all eight
  configs (its candidates die on `hold_s`, **upstream** ⇒ that 0 is *correct*). **Over-determined: step 2
  alone or step 3 alone suffices.** ⚠ Two of my claims refuted — **step 1 RELOCATES the exit** (154.4° →
  119.8°, just inside the bar) and **step 4 is not a loosening.** ⛔ **No positive control exists** ⇒ a
  merge/head-on D1 count is now evidence of **absence only.** 120 traces persisted ⇒ future detector
  questions are free. See §9.

- [x] **c97 — JUDGED, and the treatment LOST.** 4 kills / 120 against its byte-identical 41-obs control's
  **25** (same argv, same 5M budget, one variable), with the same-invocation c96 anchor reproducing the
  recorded B1 table cell-for-cell ⇒ the invocation is validated and the comparison is real. ⛔ But the
  four appended columns end at encoder-norm ranks **42/44/41/39 of 45** (0.62–1.46 vs an inherited mean
  of 4.25) ⇒ **the treatment never learned to READ them**, so c97's own hypothesis (is vertical rate
  useful?) is **UNTESTED** — it failed one stage earlier. ✅ The real result is the **control**: the
  **≥55° wall is broken for the first time in program history** (asp55 **8/20**, where c96 re-measured in
  the same invocation reads 0/20, at **4.7×** the HP per round), ⚠ at the cost of the low-aspect finish
  (asp30 19 → 4, total 45 → 25 ⇒ a **redistribution** up the hemisphere, not a gain), ⚠ the wall moved
  exactly **one rung** (asp60/75 are 0 for all three checkpoints), and ⛔ it was broken with **zero
  detectable BFM** (F8: D1 = D2 = 0.00 in all eight configs). ⚠ n = 1 run per arm ⇒ the *magnitude* is
  confounded with warm-continuation variance. See §9's c97 VERDICT.
- [x] **F8 — the POLICY-side chase D1 readings re-derived under the repaired detector. DONE, and F7's
  prediction was REFUTED.** Harness validated first (the c96 row is bit-identical to `runs/d2d/verdict.txt`).
  c95 and c87 read **D1 = 0.08/ep SHIPPED** — the first nonzero D1 events ever recorded on a learned
  policy — where c96 and c97-ctl read **0.00**. ⛔ Chasing that down found the load-bearing correction:
  the recorded *"c95's chase is D1 = D2 = D3 = 0.00"* is **c95's asp30 cell ONLY**; c95 **asp55** reads
  D1 0.50 / D3 2.50, the most detected BFM of any policy, and it kills least ⇒ **c95 was never a
  Direction-A carrier**. F8b's autopsy of all four surviving events returns **A2**: they clear the
  acceptance bar by 0.0°/0.1°, both "lethal" flags are the retracted D4 range artifact at 256/268 m with
  nothing following, and c95's two sit in a cell that dealt 0.0 HP ⇒ **still no positive control**, the
  absence-only rule stands, and **D2 has never fired on anything**. 96 policy traces persisted. See §9.
- [ ] **⚠ OPEN DECISION, for the user, created by the c97 pair — which checkpoint is the baseline now?**
  c96 (45 kills, concentrated at asp ≤ 45) and c97-ctl (25 kills, the only one that kills at asp 55) are
  **not ordered** — c97-ctl is worse on the pooled count and strictly better on the geometry the user
  actually asked for (§1: neutral head-on, *not* aspect < 90° setups). Whichever is chosen, note that the
  pair measured a **redistribution**, and that **n = 1 run per arm** ⇒ before building on the asp55 cell,
  the cheapest honest next measurement is a **second 41-obs continuation from c96 with a different seed**
  (one variable: the seed), which separates "5M more steps move kills up the hemisphere" from
  warm-continuation variance. Do **not** read the c97 treatment as evidence about vertical obs (§9).
- [ ] **The λ=100 terminal importance weight** on `reward_ce` (`agent.py:333-334`), specified by F3's
  pre-check and still unimplemented. Deferred because c96 has `mpc = False`; judge on the **greedy
  verdict**, never on predicted kill−death (tautological). Independent of the baseline decision above.

---

### Deliberately deferred

- ~~The FCS variant is a measurement in A3, not a shipped change.~~ **A3 came back negative — the
  stock FCS out-turns both variants, so there is no FCS change to consider shipping and no
  comparability to void.**

---

## 7. BFM detector spec (for A1)

⚠ **The time unit, before any threshold below.** One **control step = 41.7 ms** — `action_repeat = 5`
at the 120 Hz sim (`config.py:20`) ⇒ a **24 Hz** policy cadence. **Not 208 ms.** Three separate Phase-D
candidates made this exact 5× error and mis-sized their own interventions by 5×, so every duration in
this section is quoted in *both* units where it matters: 0.5 s = 12 steps, 1.5 s = **36** steps,
4 s = 96, 6 s = 144, 8 s = 192. The same constant fixes the discount arithmetic in §10.4
(γ = 0.995/step ⇒ 1/(1−γ) = 200 steps = **8.33 s**, not 41 s) and the world-model context in
§10.4 (H = 3 latent steps = **125 ms**, and a 4-step slice = 167 ms). Note §7's other `γ` —
`asin(v_z/|v|)` below — is the **flight path angle** and is unrelated to the discount.

One scalar carries the whole family:

- `L̂ = R_body→world · (0,0,−1)` — the lift vector (body-constant, so pure attitude).
- `φ_pp = atan2(sin(ata_az), tan(ata_el))` — the roll that puts the lift vector **on** the LOS.
  `geometry.py:80-88` already documents this expression (`body_bearing`'s `bank_err`).
- `φ_lift` = actual roll of `L̂` about `v̂`.
- **`χ = wrap_pi(φ_lift − φ_pp)`** — the **out-of-plane angle**. `χ ≈ 0` = in-plane pull (lift on
  the target); `abs(χ)` large = lift displaced off the LOS plane, which *is* out-of-plane maneuvering.

Plus `γ = asin(v_z/abs(v))` (**flight path**, never pitch attitude), `cpa_miss`/`t_cpa` from
`train_parallel.py:447 _cpa_miss_m`, closure, aspect, ata, `Es = alt + V²/2g`.

- **D1 displacement roll** — all of: (a) `abs(χ)` crosses 20° → >60° and **holds >60° for ≥0.5 s**;
  (b) `n_z > 3` throughout the hold (this clause is what makes the detector immune to the passivity
  failure mode — a wings-level drift is not a maneuver); (c) closing at entry; (d) `abs(χ)` returns
  <30° within 4 s; (e) **`sign(ata_az)` constant** across the event (a flip means you swapped sides
  = an overshoot/reversal, not a displacement); (f) `Δaspect < −15°` (it *worked*). Report
  count/episode plus `(Δaspect, Δrng, ΔEs)` per event.
- **D2 yo-yo, and the WHEN detector** — (a) `closure > 120 m/s` at entry (the genuine overshoot =
  the trigger condition); (b) `abs(Δγ)` swing ≥30° with excursion past ±15°, returning within 6 s;
  (c) `n_z > 3` during it (a pull, not a zoom-coast); (d) at exit `closure < 120` **and**
  `rng < 1500 m`; (e) **`aspect_exit ≤ aspect_entry + 10°`**. Clause (e) is the **c70 failure
  detector**: the recorded c70 pathology (⟨ata⟩ 86°→158°, dealt 27→0) is exactly "UP branch fired,
  (e) failed". Any run that raises D2 counts while (e) fails has learned to **zoom away**.
- **D3 turning-room build** — `sign(ata_az)` constant, `abs(χ) > 45°` for ≥0.5 s, `cpa_miss`
  **increasing** during the hold and **decreasing** after. The signature of the only arm that ever
  converted the merge (aspect 14°, 1 m lay) ⇒ the reference shape.
- **D4 commitment** — ⚠ **this spec was WRONG and is corrected; see §9 "D4 measured the rate
  command".** As written it was "longest run with `sign(ail)` CONSTANT ∧ `abs(ail) > 0.5`", which
  assumes `a[0]` is an aileron *deflection*. It is a normalized roll-**RATE** command
  (`f16.xml:364-396`, `fbw-override` initial 0 and never set), so a correctly flown displacement roll
  is a rate PULSE then ~0 rate to *hold* the bank — the old statistic scores it near zero and scores a
  jet spinning about its velocity vector highly. Shipped instead: **`longest_bank_hold_s`**
  (`sign(φ)` constant ∧ `|φ| > 45°`) and **`longest_chi_hold_s`** (`sign(χ)` constant ∧ `|χ| > 45°`,
  valid ticks only) — the *state* the rate command integrates to. The rate-command run is kept and
  reported as `longest_rate_cmd_run_s`, relabelled a **DITHER** measure where *low is correct*.
  Plus the `χ` histogram and the signed 1.5 s window-mean of `ail`.
- **D5 the headline number** — `P(tLeth > 0 within 8 s of an event) / P(tLeth > 0 | no event)`,
  with both base rates printed so the ratio is auditable. A policy that learned the *shape* but not
  the *when* raises D1/D2 counts at ratio 1.0; a policy that learned the *when* raises the ratio.
  Counts alone are gameable by a policy that rolls at random. Use `tLeth` (sub-8 m lay **∧**
  `rng ≤ 600 m`), **not** `tHit` — `tHit` is not range-gated and inflates.
- **D6 negative controls, mandatory** — (i) the scripted in-plane `LagPursuitTracker` must produce
  ~0 D1/D2 events (else thresholds too loose); (ii) the **fixed** `_highyoyo_stick` must produce ≥1
  clean D2/episode (else too tight); (iii) run on **c87 and c93 as well as c95** — any "improvement"
  that also shows up in c87 is a detector artifact, not a learning result.

**Never report as evidence of BFM:** episode length (does not discriminate), the train log's
`kill%`/`ret` (that is exploration, not the policy), reward-term sums (γ=1-differenced and
one-sided-gated ⇒ path-dependent), or `hypot(los_rate_az, los_rate_el)` from the existing
diagnostics (fix those four sites first — that is A4).

---

## 8. Operational notes

- **Container:** the compose service is **`dogfight-train`** (there is no `astrum-dev` service).
- **`controllers/rl/` is entirely untracked in git.** There is no HEAD to diff any RL file against.
  Do not assume `git diff` shows RL changes.
- **Always pass `--outcome-reward`** on every warm launch. It is not restored by `--resume`, is not
  in the checkpoint, and silently deletes c83/c84's gains.
- **Always `--jitter 0.5`** and always include aspect 30 in a verdict. Judge **only** the
  out-of-band greedy verdict, never the train log's `kill%`/`ret`.
- **Verify `collected ≈ 5M`** in every verdict header before believing it.
- Tests: `make env-test` (Rust/env), `make rl-test` (RL). `make lint`'s baseline is 131 pre-existing
  ruff errors, all in `_*_tmp.py`/`_diag_*.py` throwaways, and `ruff format --check` already fails
  on `train_parallel.py` (pre-existing aligned-comment blocks). **Do not reformat that file** — it
  is churn in a mid-experiment file.
- Sim (Rust) tests go in `tests/`, not inline in `main.rs`. Prefer Makefile targets over shell
  wrapper scripts.

---

## 9. Results log

### D3b REPLAY AUTOPSY — the kill is a head-on face shot, the detector is sound, and three of my own reported facts are RETRACTED (2026-09-06)

Harness: `controllers/rl/_diag_d3b_replay.py` (re-flight + 88-column per-step dump),
`_diag_d3b_detector_audit.py`. Artifacts: `runs/d3b/replay_kill_seed62.csv`,
`replay_nokill_seed28.csv`, `replay_report.txt`, `detector_audit.txt`, `detector_audit2.txt`.
93 claims were raised across six lenses and adversarially verified one-per-claim; **77 were refuted or
narrowed, 16 survived intact.** (The verifier prompt defaulted to `refuted` on any number it could not
independently reproduce, so many "refutations" are tightenings — read each corrected claim, not the tally.)

**Reproduction is exact.** **Seed 62** is the unique source of `d3b_best.npz`: `max|Δ| = 0.000e+00` on the
full `(173,5)` actions *and* the `(173,41)` obs. Actions are **bit-identical across all 8 seeds** — the
controller reads own attitude only, so gun damage cannot perturb own dynamics; seeds change only *when* the
100th HP lands. The 172-vs-173 / 0.458-vs-0.500 discrepancy is the npz save rule
(`max(rows, key=(kill, tleth_s))`, Python first-maximal tie-break), not an artifact defect.

**Only 2 of 6 segments ever flew** (seg 0 in full, seg 1 truncated at 80% by the kill). The saved "24 s
trajectory" is a **7.2 s, two-segment maneuver expressing 6 of 18 genome dimensions.** Seg 0: +115° bank
(past vertical), full aft stick, 5.73 g sustained, α peak 12.28° of a ~16.7° ceiling. Seg 0→1 boundary
produces a **0.125 s, 7.38 g roll-coupled transient at *reduced* stick** — so `nzMx 7.4` is not a sustained
pull. Seg 1: unload to 48° bank, full afterburner, nose-low run-in.

#### What the maneuver is

**A nose-low, over-banked, continuously descending nose-to-nose closing turn into a point-blank
near-head-on face shot — and it is NOT a lead turn.** The advocate lens named it a "lead turn for the gun";
verification refuted the naming: **HCA moves 1.48° in 6.5 s** (−180.000° → −178.521° at first fire). Aspect
is **147.6–176.1° on all 173 steps, 0 of 173 below 90°** — it never enters the bandit's rear hemisphere.
Range falls monotonically 3055 → 8.4 m at +373..+478 m/s closure (≈ the two airspeeds added). There is no
pass and no circle: **the bandit dies at the merge point**, so the one-circle/two-circle taxonomy does not
apply. The pull is **lag pursuit**; the surviving mechanism is that **range collapse shrinks the required
lead as time-of-flight → 0** (`alpha_req` 45.5° → 10.1°), not that anyone was out-turned. All 100 HP arrive
in **10 control steps = 0.417 s, inside 186 m**; `lay_miss` is a clean ordinal hit predictor (16 rounds
above 8.0 m → 0 HP, 50 below → 50 hits). ⚠ The sim has **no aircraft–aircraft collision model** (bullet↔
sphere only, `combat/systems.rs:121-145`) and post-terminal range is **5.36 m** = airframe interpenetration;
it is still a real gun kill in the sim's accounting (62 of 100 HP landed at ≥ 67 m) but the hits that
crossed zero were inside 47 m.

#### ⛔ It is a knife-edge alignment, not a maneuver

Shifting the genome's own segment clock by **±83 ms** (±2 steps of a 24 s horizon): **5/8 kills, 99.0 HP →
0/8, 0.0 HP**. **10 of 10** perturbed start geometries: **0/8 kills, exactly 0.0 HP**, with
time-with-a-valid-gun-solution collapsing to **0.000 s** even though each arm still spends 1.9–2.7 s inside
600 m. Across **34 perturbed arms × 8 seeds: 33 arms zero kills, 31 exactly 0.0 HP** — and in every failure
the jets still pass within 10–40 m. On the unperturbed arm, kill-vs-no-kill is decided by **gun dispersion**
(96–100 HP on every seed; the three misses are one or two 2-HP hits short). **The fire discipline is a
hand-coded oracle — the 18 optimised numbers never chose when to shoot.** ⇒ **F1 is dead** (§6).

#### Why the T3 lost, and it was not flying badly

DEFENSIVE from step 36 (79% of the episode), holding a textbook **5.3 g / 84° break while CLIMBING +103 m
and gaining Es +143 m**. It loses because **`BREAK_BANK_RAD = 85°` is a hard constant with no LOS-plane
term** (`constants.py:123`, `opponent.py:425-461`) while the **+2000 ft perch tilts the LOS plane**:
`los_rate_el` +6.01°/s against `los_rate_az` +3.09°/s, and an 84° bank puts only ~10% of its turn rate into
elevation (ω_v 1.28 vs our 6.78 °/s). **It out-turns us by 27% in total rate (12.99 vs 10.23 °/s) and still
loses the aim.** It also fired **98 rounds for 2 hits** — emptying at 500–945 m where p_hit ≈ 0, then losing
its 4° gate exactly as p_hit → 1, which is `gun-lethal-envelope-600m` recurring on the opponent side.
**Untested falsifiable prediction (worth one cheap run): `me_dalt_ft = 0` should remove this kill.**

#### ⛔⛔ Three retractions of things this document and I previously asserted

1. **"0 D1/D2 in 15,187 rollouts" — RETRACTED, the statistic never existed.** The detector was **never
   evaluated inside the search loop**; it only ever saw **8 episodes per arm** (the best genome re-flown).
   Never again cite a detector count over a rollout budget.
2. **"chiMx 178.7° and nzMx 7.4 g ⇒ both out-of-plane ingredients were present on the winner" —
   RETRACTED.** That row is a **pooled sum over 8 episodes** and the two numbers are from different
   flights. On the kill trace max|χ| = **112.1° at step 0, t = 0.000 s, nz 0.57 g** — the *spawn attitude,
   before any control input took effect*; max|χ| among **loaded** ticks is **60.5°**; loaded-and-out-of-plane
   co-occur on **3 of 173 ticks**. The 178.7° belongs to **seed 28, step 410, at nz = −0.89 g** (negative g).
   And **D3 = 3 pooled is D3 = 0 on all five kill seeds.** ⇒ **never read a pooled detector row as
   per-trajectory evidence.**
3. **"The 4.0 s parameterisation may not have been able to express a displacement roll" — REFUTED.** A
   concrete 18-dim genome **inside the exact search space** scores **D1 = 1 on all 8 seeds with a 2.583 s
   loaded hold (5.2× the 0.5 s bar)** — and is worth **+1.881 against the incumbent's +20,007.996**. χ is
   not commanded: under a constant bank target it drifts at +20.2°/s mean, so one 4.0 s segment sweeps the
   whole 40° D1 span unaided. Corroboration: the two hand-coded **closed-loop** arms also score D1 = D2 = 0
   on this cell. ⇒ **the no-BFM outcome measures the OBJECTIVE, not the parameterisation.** ⚠ D2/D2c *are*
   foreclosed by the **cell** at every resolution from 4.0 s to 0.5 s (entry gate opens once at tick 0;
   `clean` needs closure < 120 m/s but the minimum on the trace is +373 m/s; the elevator channel is
   pull-only) — so finer segments are not the fix there either.

#### Detector verdict: sound on strictness, three real scope limits

**Not too strict.** D1 = 0 across **all 20** combinations of `CHI_OUT_DEG` ∈ {60,50,45,40,30} ×
`CHI_MIN_HOLD_S` ∈ {0.5,0.3,0.2,0.125}; loosening lengthens a single contiguous hold (0.083 → 0.833 →
1.583 → 2.250 s) rather than un-fragmenting several. D2c = 0 in all 18 combos. Trace fidelity 1.688e-14.

- **Limit 1 — D1/D2 define success as a REAR-QUARTER conversion.** `bfm_detect.py:373-374` requires
  `d_aspect <= -15.0°`; the best achievable on a head-on gun pass (excluding the fly-through tick) is
  **−6.7°**. ⇒ **"D1 = D2 = 0 on a head-on kill" licenses "no displacement roll, no yo-yo, no
  turning-room build" and does NOT license "no BFM."**
- **Limit 2 — coverage: there is no lead-turn detector.** `detect_all` (`bfm_detect.py:738-755`) has
  exactly three shape detectors, and a lead-turn detector cannot be added by loosening a threshold
  because its acceptance clause is the **negation** of D3's and its aspect clause the negation of D1/D2's.
- **Limit 3 — D3 is TOO LOOSE, and it is the one criterion that is actually wrong.** No magnitude floor on
  the build clause ⇒ the single D3 positive in the whole 8-seed pool is a **12.4 m (0.41%) CPA build during
  a 539 m egress.** Fix before citing D3 again.
- Minor: every hold statistic understates a K-tick run by **one control step (41.667 ms)**.

⚠ **What the audit does NOT establish.** It ran on **one open-loop trajectory pair** and shows only that the
detector does not FALSELY REJECT it — **never** that it would ACCEPT a real maneuver. **The positive control
was not run and is the cheapest high-value follow-up: re-fly a `bfm.py` lead-turn / yo-yo primitive on
merge1500 and run `detect_all` on it.** It also does **not** retroactively validate the D1 = D2 = 0.00
readings on the c87/c93/c95 **policy** checkpoints — different closure profiles, longer episodes,
closed-loop control, and no checkpoint was loaded.

#### New defect found (not on the critical path, but real)

**The search's fire gate can fire REARWARD.** `lay_miss_m` uses `sin(lead_error)` with `lead_error` an
unsigned `[0, π]` angle, so a bandit directly **astern** aliases to one directly **ahead**. It never touches
the kill (which is nose-on throughout) but it burned rounds on the non-kill seeds.

#### Consequence for §10 and for the three-instance synthesis

**Instance 3 is now the WEAKEST of the three, not the strongest, and it is re-attributed.** It is one
brittle alignment on one fixed start, its headline detector statistic did not exist, and a
detector-passing displacement roll demonstrably lives inside the same search space at 1/10,600th of the
fitness. **The admissible claim is "objective J does not require BFM, and a kill exists that does not use
it" — NOT "nothing in this environment needs BFM to score."** Fitness ranking is not an environment fact.
Instances (1) c95's chase at D1 = D2 = D3 = 0.00 and (2) c96's 45 kills with zero BFM are untouched and
remain the load-bearing evidence — and both are **learner-side**.

Append measured results here as each checklist item lands. **Numbers only go in this section after
they have actually been run** — do not pre-fill expected values.

### E2b PRE-CHECK — the termination head's PREDICTION is consumed NOWHERE, so E2b's prize is auxiliary-only (2026-09-06)

Free, two commands, read straight off the shipped checkpoints. It reshapes E2b before a single step of
training was spent on it, and it corrects two things this document and the memory record assert.

**Measured**, `runs/curriculum-96/run.pt` (collected 5,000,847) and `runs/curriculum-97-white/run.pt`
(collected 4,814,165), identical on both: `mpc = False`, `episodic = True`, `horizon = 3`,
`termination_coef = 1.0`, `rho = 0.5`, `num_samples = 512`, `num_pi_trajs = 24`.

`train_parallel.py:4635` is the cause: **`args.no_mpc = not args.collector_mpc`**, and `--collector-mpc`
appears in no launch line in this program. So `cfg.mpc` is False in every parallel run, and:

1. **`_estimate_value` is unreachable.** Its termination latch (`agent.py:290`,
   `dead = (termination(z) > 0.0)`) is called only from `plan()`, and `plan()` only from `act()` under
   `if self.cfg.mpc` (`agent.py:203`). ⇒ the latch has never influenced one action.
2. **`_td_target` does not use the head either** — it takes `terminated` from `buffer.sample()`
   (`agent.py:299-304`), i.e. the ground-truth flag, not a prediction.
3. ⇒ **the termination head's only channel is the auxiliary BCE gradient into the shared latent.**

**⚠ Scope of this entry, stated honestly: `cfg.mpc = False` is NOT a new discovery.** It is already on
record (the 2026-09-04 planner audit measured it in three checkpoints, and flagged two *other* planner
defects as "inert **only** because `cfg.mpc=False`"). What is new here is (a) the confirmation carries
forward to c96 and c97, and (b) **the one place the existing record did not apply its own `mpc=False`
caveat is the termination head** — so:

- **E2a's headline prize is downgraded, not deleted, and it is E2a's write-up that needs the edit.** E2a
  presented the ×3 positive-rate lift as "the first mechanism anyone has offered for the dead termination
  head". The *data* claim stands exactly as measured (1 positive per 3,305 BCE elements, base rate
  3.03e-4). The *mechanism* claim does not: with `mpc=False` nothing reads the head's output at decision
  time, and `_td_target` uses the buffer's ground-truth `terminated`, so the head's only route to
  behaviour is the auxiliary BCE gradient into the shared latent.
- **What this does NOT retire.** The recorded conclusion "disengaging is the value-maximising action" does
  not depend on the head at all — it rests on every fight state having negative Q (mean −66.5) against
  `R_BUBBLE_EXIT == R_TIMEOUT == −50`. That argument is untouched. Likewise the "MPPI horizon = 125 ms"
  note was already accompanied by "the planner is irrelevant as configured"; this entry does not overturn
  it, it just stops the horizon number from being quoted as a live limiter.
- **No past verdict is contaminated:** `_bfm_verdict_tmp.py` defaults to `--modes greedy` ("the out-of-band
  read"), which matches how the agent was trained. Checked, not assumed.

**Net effect on E2b:** its case now rests entirely on (a) the terminal's share of the *reward/value*
supervision, 0.013% → 0.091%, and (b) a ×3 auxiliary termination signal whose only route to behaviour is
representation shaping. That is a **small-share** fix and must be written up as one. Form (i)
(oversample terminal-hosting starts) remains the cheapest and is the only one whose benefit survives this
pre-check intact; forms (ii) and (iii) buy weight and depth for a quantity nothing consumes directly.
**Open, and the thing to measure before spending a run:** does the head discriminate at all
(AUC on real terminals) and is it calibrated to the 3.03e-4 base rate? If it already discriminates, the
auxiliary signal is not starved and E2b's remaining half is only (a).

### ⚠ OPS DEFECT, FIXED — `make rl-env` silently SIGBUSes every running collector and worker (2026-09-06)

**Not a BFM result; a repo footgun that destroyed two live runs in one minute and would have kept doing
it.** `Makefile:79-83` staged the extension with `cp target/release/libastrum_env.so
controllers/rl/vendor/astrum_env.so`. `cp` opens the **existing inode** `O_TRUNC` and rewrites its pages
in place. Every collector process and every diagnostic worker has that `.so` **mmap'd**, so its text
segment changes underneath them and they take **SIGBUS/SIGSEGV — with no Python traceback, and all of
them at once**. And `rl-tdmpc2-train-parallel`, `rl-smoke`, and both smoke targets all declare `rl-env`
as a prerequisite, so **launching any `make rl-*` target kills every run already in flight.**

Measured timeline (host mtimes, one minute):

| time (EDT) | event |
|---|---|
| 05:20:12.763 | D3b search container starts; 6 spawn workers mmap `vendor/astrum_env.so` |
| ~05:23 | D3b completes gen 1 (85 s/gen, healthy) |
| **05:23:51.250** | **`vendor/astrum_env.so` rewritten in place** by an Arm-B `make rl-tdmpc2-train-parallel` → `rl-env` |
| 05:23:51.711 | Arm B's trainer container starts (maps the *new* inode, unaffected) |
| 05:23:55.612 | Arm A saves `runs/curriculum-97-white/run.pt` at **4,814,165 / 5,000,000** steps after `all 6 collectors exited` |
| 05:23:59.205 | Arm A's final log write; container exit **139** (128+11 = SIGSEGV) |
| 05:24 onward | D3b never produces gen 2; parent at **1.9% CPU**, all 6 workers gone, **zero** bytes of diagnostic output |

**D3b's kill is established** — its workers held the old inode across the rewrite, died in that second,
and the raw stream contains no `Traceback`, no `Error`, no `panicked`, nothing (signal death is silent by
construction). **Arm A's cause is NOT established and must not be written down as if it were:** its
collectors dying at 05:23:51 and the parent noticing 4 s later fits perfectly, but the `make` that did the
rewrite was itself gated on Arm A's pid having already disappeared, and both readings cannot be true.
Recorded as: *Arm A ended at 4,814,165 steps (96.3% of budget) with all 6 collectors gone, no traceback,
exit 139, no OOM evidence (125 GB RAM, 102 GB available, load 9.7/24) — same signature as the confirmed
D3b kill, ordering unresolved.*

**Two fixes shipped:**
1. `Makefile:79` now stages via a same-directory `mv` (`cp` to `.astrum_env.so.new`, then `mv -f`). A
   rename swaps the **name** to a new inode; processes holding the old one keep it mapped and alive. The
   comment in the recipe records why, so it does not regress to `cp`.
2. `_diag_true_mpc.py`'s `Evaluator._collect` replaces a bare `out_q.get()` with a 30 s poll that asks the
   OS whether the workers still exist and **aborts loudly** with each worker's exitcode (negative =
   signal) instead of blocking forever. The old code is why a dead search sat there *looking* alive.

**Standing rule this earns:** never launch a `make rl-*` target while a trainer or a search is live, and
when a long run dies with no traceback, **check `stat` on `controllers/rl/vendor/astrum_env.so` before
theorising**. Also: `docker rm -f` the search's container after killing the host-side `docker compose`
CLI — killing the CLI leaves the container `running`.

### E2a — the replay sampler DOES starve the episode-final transition. CONFIRMED at exactly 7.000× (2026-09-06)

`controllers/rl/_diag_e2a_buffer.py`, CPU-only, deterministic (`SEED = 20260906`), run against a **real,
full 1,000,000-slot `ReplayBuffer`** filled through the shipped `add_episode` with the live run's measured
episode-length mix (mean **692.9** steps, from `runs/curriculum-97-white/train.log` windowed over 2,474
episodes; clock cap 1,440 = `--max-time-s 60` at 24 Hz). Geometry read from `runs/curriculum-96/run.pt`'s
own cfg: `horizon 3, rho 0.5, batch 256, buffer_size 1e6, episodic True`.

**M0 — exact enumeration of all 995,689 valid starts (no RNG). The full ρ-weight profile:**

| position in episode | n | ρ-weight | reach | hits at t = 0/1/2 |
|---|---|---|---|---|
| 1 (first transition) | 1,388 | **1.0000** | 1 | 1388 / 0 / 0 |
| 2 | 1,388 | 1.5000 | 2 | 1388 / 1388 / 0 |
| 3 … T-2 | 991,163 | **1.7500** | 3 | all three |
| T-1 | 1,389 | 0.7500 | 2 | 0 / 1389 / 1389 |
| **T (episode-final)** | 1,389 | **0.2500** | **1** | **0 / 0 / 1389** |

⇒ **interior : final = 1.7500 / 0.2500 = 7.000×**, closed-form, not 6.82×. D5's 6.82 was a Monte-Carlo
estimate of a quantity with an exact value. **M6**: the ratio is 7.000 at every episode length tested
(20/50/100/200/693/1440) ⇒ it is a property of the slice geometry, not of the run's episode mix.

**The claim that matters is the depth, not the weight.** The final index is reachable from exactly one
start and always lands at `t = 2`: **0 of 904** enumerated hits and **0 of 4,648** occurrences read straight
off the shipped `buffer.sample()` tensor are at `t < 2`. So the kill reward is supervised only at `zs[2]`,
a latent two dynamics steps downstream of the encoder, and never at `zs[0]`.

**The negative control did the work it was there for.** The episode's FIRST transition is *also* reach-1 —
but at `t = 0`, weight 1.0, ratio only **1.75×**. So the finding is **not** "boundary indices are
down-weighted" (which would have been a much weaker claim); it is specifically that the *episode-final*
index is pinned at the most-discounted ρ position. Had the control not been run, the two were
indistinguishable.

**Sampler uniformity (M1b), so the profile can't be blamed on sampling bias:** over 49.8 M draws
(λ = 50.0 per valid start), mean draws per start by start class — `first` slot 49.703 (z −1.59), episode
head 50.026 (z +0.25), interior 50.000 (z −0.06), episode tail 50.177 (z +1.65), the terminal-hosting start
`T-3` 50.309 (z +1.66); **0 invalid starts drawn, ever**. `_sample_starts` is uniform over valid starts, so
the deficit is pure slice arithmetic — and therefore fixable in the sampler.

**Base rates, which must be quoted with the ratio (M5/M3/M4):**

| quantity | shipped | if the final index were treated as interior |
|---|---|---|
| share of the reward/value ρ-weight | **0.013%** | 0.091% |
| terminal transitions per 256-slice batch | **0.232** (1 every **4.3 updates**) | ~0.7 |
| `terminated` positive rate in the (H,B) tensor | **3.03e-4** (1 per **3,305** BCE elements) | 9.08e-4 |

⇒ this is a 7× on a term that is **1/8000 of the loss**. K4 did not fire, so "starved" is a fair word, but
the §9 rule that killed D5's "lift" applies here too: **never quote the ratio without the base rate.**
⚠ **D6a partly blunts the depth argument** — the dynamics model is accurate (cos 0.926–0.963 at depth 30),
so `zs[2] ≈ encode(obs[2])` and "never supervised at depth 0" is milder than it sounds.

**Finding E2a turned up that D5 did not claim, and that no prior box had a mechanism for:**
`termination_loss` (`agent.py:337-341`) is a plain `binary_cross_entropy_with_logits` **mean** — it is
**not** ρ-weighted. So for the termination head the defect is not the weight at all, it is purely the
reachability *count* (1 instead of H): the head sees **one positive per 3,305 elements**, and E2b would ×3
that. This is a candidate common cause for the recorded **dead termination head**
(§10.4 / `tdmpc2-credit-horizon-ceiling`).

**Design consequence that closes E2b's cheapest option as written:** the final index can only reach `t = 0`
from start `s = T-1`, whose window spans `T-1, T, T+1, T+2` — it **straddles the episode boundary**. So
"allow the terminal at `t=0`" is not a one-line relaxation of the `_first` rejection; positions past the
terminal would train on the next episode. E2b needs either a post-terminal validity **mask** threaded into
`agent.update`'s three ρ-weighted sums, or terminal-slice **re-weighting/oversampling** instead of
relocation. See the E2b box for the three ranked forms.

**⚠ Method finding: a pre-registered threshold below its own estimator's noise floor is not a test.**
K3 ("M1 disagrees with M0 by > 2% on the ratio") **fired on the first run at 21.10%** — and it was an
artifact of my own estimator. A boundary class has only ~1,389 members with reach 1, so at the first pass's
λ = 0.10 draws per valid start its Monte-Carlo SD was ~8% — four times the threshold it was being tested
against. All five boundary classes came out low together because adjacent tail positions **share** hosting
starts, so the errors are strongly positively correlated, which is what made noise look systematic. Two
fixes, both now in the harness: raise λ to 50 (per-class rel err drops to 0.00–0.71%, all |z| ≤ 2.00, ratio
6.951× vs 7.000×), and add **K3′**, the same question asked where the count is large — pool all terminals
and read the rate off the shipped `sample()` tensor: enumerated **9.079140e-4** vs shipped **9.078125e-4**,
disagreement **0.01%**, z = **−0.01**. *Pre-register the threshold against the estimator's noise floor, not
against what would be interesting to detect.*

**Gate: 0 of 6 fired.** K1 (reachable at t<2) no — 0/904 and 0/4,648. K2 (ratio < 2.0) no — 7.000×.
K3 no (0.70% at λ=50) and K3′ no (0.01%). K4 (share ≥ 1%) no — 0.013%. K5 (≥ 1 terminal/batch) no — 0.232.

### E1a — the decoder does NOT compress the kill. FAIL, and it retracts D5's mechanism (2026-09-06)

`controllers/rl/_diag_e1a_decode.py`, one invocation: **c96 (RUN) / c95 / c87 (CONTROLS) × chase asp30
and asp75 × 6 greedy episodes**, `--jitter 0.5`, CPU (so it did not contend with the D2d run on the GPU).
Reuses `_diag_credit_reach.py` read-only for env construction and cell definitions. Headers verified:
c96 `collected = 5,000,847`, `outcome_mode=True`, γ=0.995, bins `[-10,10]×101`, `num_q=5`, `mpc=False`.
asp75 is in the set on purpose: c96 kills **0/20** there, so K3 has a spike-free comparison *inside the
same invocation*.

**Three of the four pre-registered kill conditions fired. E1b is closed.**

**M0 — the codec is EXACT** (K1 does not fire). `two_hot_inv(log(two_hot(x)))` over
x ∈ {0, ±1, 8, 35.4, ±50, −60, 147.41, 150, ±560, 3000}: worst **relative** error **1.43e-6**. So any
gap must come from the *predicted* distribution being spread over non-adjacent bins.
**M1 — no clipping.** The largest probe, symlog(3000) = 8.01, is inside `vmax = 10`; symlog(R_KILL=150)
= 5.02 and symlog(|J|≈560) = 6.33. The bin axis has headroom.

**M2 — and the predicted distribution is NOT spread.** `N_eff bins = exp(entropy)` is **1.25 / 1.83 /
2.63** (p10/p50/p90) on the kill cell — i.e. essentially two adjacent bins, *the exact regime M0 proves
exact*. Consequently on c96/asp30 (6/6 kills, 745 states):

| quantity | p10 | p50 | p90 | max |
|---|---|---|---|---|
| `q_ship = symexp(E_p[bin])` (shipped) | 14.95 | **41.12** | 53.40 | 55.73 |
| `q_true = E_p[symexp(bin)]` (Bellman) | 15.06 | **41.29** | 53.43 | 55.93 |
| `Δ = q_true − q_ship` | 0.034 | **0.140** | 0.241 | 1.659 |
| modal bin (scalar) | 15.44 | **43.70** | 53.60 | 53.60 |

**D5's "median +35.4 vs a modal bin of +147.41" does not reproduce.** The modal bin is 43.70 (p50) /
53.60 (p90) — never 147.41, which is exactly `symexp(5.0)`, i.e. bin index 75 on the nose. The one number
that *does* partly reproduce is mass above +50 (mean 0.211, p90 0.972, and p50 0.406 in the 2–4 s bucket),
but it is mass just *above* +50, not at +147: `q_ship` in that bucket is 47.47 and `q_true` is 47.71. The
decode gap is **0.3–0.5% of |Q| on all six cells**, not 4×.
**Self-check:** `|q_ship − Q(...,"avg")|max = 0.00e+00` in every cell — the harness's decode path is
bit-identical to the shipped one, so this is not a re-implementation artifact.

**M4 — K2 FIRES on every cell. This is the decisive one.** `symexp` is monotone, so at a *fixed* state the
decode cannot change `argmax_a`; it can only do work by re-ordering states into `_td_target`. It does not:

| ckpt / cell | n | kills | rel Δ p50 | Δ p50 | massHi p50 | Spearman ρ | discordant pairs |
|---|---|---|---|---|---|---|---|
| c96 / asp30 | 745 | **6** | **+0.0036** | +0.140 | 0.0012 | 0.99999 | 0.065% |
| c96 / asp75 | 8640 | 0 | −0.0028 | −0.121 | 0.0000 | 0.99999 | 0.022% |
| c95 / asp30 | 8640 | 0 | −0.0014 | −0.028 | 0.0000 | 1.00000 | 0.017% |
| c95 / asp75 | 8640 | 0 | −0.0019 | −0.071 | 0.0000 | 1.00000 | 0.006% |
| c87 / asp30 | 8640 | 0 | −0.0045 | −0.186 | 0.0000 | 0.99998 | 0.130% |
| c87 / asp75 | 8330 | 0 | −0.0028 | −0.117 | 0.0000 | 0.99999 | 0.059% |

Top-decile-by-`q_true` mean rank shift under `q_ship`: **≤ 0.07%** of ranks everywhere.
**M5/K3 also FIRES:** relative Δ is +0.0036 on the one kill-bearing cell against −0.0014…−0.0045 on the
five spike-free ones — same order of magnitude, and the sign merely tracks `sign(Q)` (Jensen: `symexp` is
convex above 0 and concave below). Δ is a ~0.4% *multiplicative* rescale of the value scale, and the symlog
two-hot head exists precisely to be robust to value scale.
**K4 quiet:** tail share **0.0000** everywhere, so the raw expectation would in fact have been numerically
safe (`symexp(±10) = ±22025`, so this was worth checking) — but that is moot now.

⚠ **Do not cite M6's `Δ/(1−γ)` line as a magnitude.** It prints −23.3 to −37.3, which is only the right
compounding if Δ were a constant per-backup offset. Δ's sign is locked to `sign(Q)`, so it is a rescale,
not an offset, and it does not compound that way. The line is a designed-in worst case, not a finding.

**The method lesson, and it is the important part.** Both of D5's "structurally new levers" came from
**agent-reported numbers with no committed harness in the repo** — `grep` for `147.41` or the mass figure
finds nothing outside E1a's own docstring quoting them. E1a is the first committed measurement of either
and it contradicted the one it tested by ~500× on the mass figure and ~250× on Δ. ⇒ **an unreproducible
statistic is not a mechanism.** §6 Phase E now says to run **E2a before believing E2's 6.82×** for the same
reason; E2 is untouched by this result because it is an arithmetic property of `buffer._sample_starts`,
independent of the decode.

### A1 — the BFM detector exists (2026-09-05)

`crates/astrum-env/python/astrum_dogfight/bfm_detect.py`, 724 lines, D1–D6 per §7. 20 unit tests,
wired into `make env-test` (88 tests green there in total). Nothing else in §9 would be measurable
without it, and D-D is now closed.

### A4 — the five measurement bugs, fixed at the root (2026-09-05)

The `hypot(los_rate_az, los_rate_el)` bug had recurred four times for one reason: **the correct
scalar did not exist anywhere**, so every caller re-derived it wrongly. Fixed by adding it —
`RelGeom.los_rate = |v_perp| / rng`, the total LOS rotation rate (`geometry.py`) — and repointing the
four sites at it. Pinned by `test_total_los_rate_is_not_the_hypot_of_its_body_components`, which
asserts the **exact** relation rather than a bound:

> in a coplanar crossing, `hypot(los_rate_az, los_rate_el) == los_rate · cos(ata)`.

`v_perp ⟂ LOS` forces its nose and starboard components into the ratio `−tan(ata)`, so
`|v_perp| = |v_perp·starboard| / cos(ata)`. The understatement factor **is** `cos(ata)` — worst
exactly at the high-ata merges this program studies. At the test's 60° the old statistic reported
**half** the true rate.

The `a[4] >= 0.5` trigger threshold was in **five** files, not the one recorded here:
`_energy_merge_probe_tmp.py:105`, `_diag_trigger_tmp.py:69`, `_lead_probe_tmp.py:91`,
`_diag_headon_alt_tmp.py:102`, `_merge_trace_tmp.py:89`. All now `>= 0.0` with the contract quoted
inline.

### A2 — the demonstration library is hollow (2026-09-05) ⇒ **D-A is not a lever, and C1 is closed**

`_diag_stick_ceiling.py --mode ceiling`, 5 arms × 2 throttle variants × 4 episodes = 40 episodes.
Neutral head-on vs the **fixed** T3, sep 3000 m, lateral ±1400 m, 180 s, jitter 0.5, stick flying
**100% of ticks** (so defect 1's coin and the splice window are absent by construction — this is each
stick's own ceiling).

| arm | nz p50 | TAS p50 | D1/ep | D2/ep | lethal ticks | best lay | best aspect | ends |
|---|---|---|---|---|---|---|---|---|
| inplane (control) | 5.26 | 222 | 0.00 | 0.00 | **0** | **1.9 m** | **2.4°** | hard deck 42 s |
| leadturn (= c95) | 4.96 | 251 | 0.00 | 0.00 | **0** | 3.9 m | 12.7° | timeout 180 s |
| lowyoyo | 2.12 | 354 | 0.00 | 0.00 | **0** | 154 m | 16.8° | hard deck 82 s |
| highyoyo | 1.21 | 330 | 0.00 | 0.00 | **0** | 160 m | 76.7° | left bubble 52 s |
| egg | 0.73 | 150 | 0.00 | 0.00 | **0** | 9.5 m | 62.6° | 13.6 km, timeout |

**Zero kills, zero rounds, zero lethal ticks in all 40 episodes.**

1. **D2 = 0.00 for every arm at every threshold** — the sensitivity sweep relaxes `NZ_LOADED`
   3.0 → 0.5 g (below which wings-level cruise would itself count) and `CHI_OUT_DEG` 60 → 30°, and
   D2 stays 0.00 everywhere. D2 keys on closure and the flight-path swing, so the load clause is not
   what rejects it: **`_highyoyo_stick` does not fly a high yo-yo shape at all.** Not a detector
   artifact — this is the reading D6 control (ii) demanded, and it comes out "hollow stick", not
   "too-tight detector".
2. **`highyoyo` and `lowyoyo` score 0.00 on D1/D2/D3 at *every* setting**, including
   everything-relaxed. `highyoyo`'s `|chi|` median is 51° — it *is* banked off-plane — but never
   coherently at load, and never with the return-to-plane that makes a roll a maneuver. At nz p50
   1.21 and 330 m/s (corner ≈ 225) it is a **banked coast**, which is exactly the passivity failure
   mode D1/D2 clause (b) exists to reject.
3. **The detector is not structurally dead**: `leadturn` goes 0.00 → **1.00** D1/ep and `inplane`
   0.00 → 0.25 when only the chi bar drops to 30°, with the 3 g load clause untouched. So the lead
   turn does carry **loaded out-of-plane content — but shallow, in the 30–45° band**, under the
   doctrinal 60° bar. It is the one arm in the library with real content, and the only one that
   survives 180 s. **Do not loosen the shipped 60°** to manufacture counts; that is the D6 (iii)
   artifact trap. Record the curve instead.
4. **Defect 2 (rudder) REFUTED by measurement.** `|beta|` on committed-roll ticks is p50 1.8–2.6°,
   p90 2.9–5.7°, max 9.9° across all five arms. The FLCS Yaw channel (`f16.xml:660-772`:
   `fcs/yaw-load-norm` = 0.25 × `n-pilot-y-norm`, summed with yaw rate into `fcs/yaw-load-pid` →
   `fcs/yaw-scheduler` → rudder) is an automatic turn coordinator. **Rudder 0.0 is correct. Do not
   add rudder to the sticks.**
5. **Defect 3 (throttle units) is real in source but NOT load-bearing.** For `highyoyo`/`lowyoyo`
   native throttle p50 stays **1.00** under both `--thr asflown` and `--thr fixed`, and every
   geometry number is identical to 0.1 m — they never reach the branch carrying the sub-1.0 constant.
   So "the bleed half of every yo-yo runs at double power ⇒ a high yo-yo that cannot bleed is a zoom"
   is refuted **as the cause**: the high yo-yo is a zoom because it is not pulling (nz p50 1.21), not
   because of its throttle. The fix's only measurable effect is on the closed-loop tracker:
   TAS 252 → 222 (toward corner 225), lay 8.5 → **1.9 m**, aspect 8.4 → **2.4°**.
6. **Defect 1 (per-step coin) is moot for shape.** At 100% stick duty the arms already fly no
   D1/D2, so restoring a per-episode coin cannot create a maneuver that is not there. `--mode splice`
   was therefore **not run**: it can only lower a ceiling that is already 0.00.
7. **New defect: the `egg` stick is broken.** nz p50 0.73, TAS 150 m/s, **alt p50 13 619 m**
   (≈44 700 ft) — an unloaded full-power zoom to the stratosphere that burns 164 s per episode. It is
   not a maneuver and should not be spliced. (Not the active mode; c95 used `leadturn`.)
8. **D-E corroborated from an independent direction.** `inplane` reaches a **1.9 m** lay and
   `leadturn` a **3.9 m** lay with **zero lethal ticks** — those lays land outside 600 m. Measured
   with scripted sticks, no checkpoint involved, so it is not a policy artifact: **lay accuracy is
   not the wall; range at the moment of solution is.** This is B1's case, made twice over.

### A3 — the FCS is not the limiter: **D-C is REFUTED** (2026-09-05)

`make jsbsim-variants` + `_diag_turnperf.py --model {f16,f16nopid,f16kp10} --banks 85`, all three run
from the **same** copied JSBSim root so the XML edit is the only difference between arms. Max-
performance pull (85° bank, `elev = −1`, throttle 1). Sustained = t = 10 s.

Sustained Nz / turn rate (°/s) / radius (m):

| alt | TAS | stock f16 | f16nopid (trigger removed) | f16kp10 (kp 0.3→1.0) |
|---|---|---|---|---|
| 5 kft | 180 | **6.03** / 15.7 / 794 | 5.57 / 14.0 / 930 | 5.70 / 12.8 / 1130 |
| 5 kft | 260 | **7.84** / 17.8 / 798 | 6.96 / 15.0 / 1001 | 6.52 / 12.7 / 1299 |
| 15 kft | 180 | **4.72** / 12.8 / 948 | 4.43 / 11.7 / 1066 | 4.83 / 11.3 / 1232 |
| 15 kft | 260 | **6.75** / 15.4 / 923 | 6.09 / 13.1 / 1148 | 5.64 / 11.3 / 1442 |

**The stock FCS is the best of the three, everywhere.** Ungating the integrator *lowers* sustained g
by 0.3–0.9 and grows the turn radius 15–25%; raising kp to 1.0 is worse still (radius +42% at
5 kft/260). Both variants also settle at **lower alpha** than stock (8.6° and 5.7° vs 10.6° at
5 kft/180) — the modified loops stabilise on a weaker pull, they do not "release" a suppressed one.
So the premise "every jet has been flying at ~half its available g because a `<trigger>` freezes the
ki = 0.025 integrator" is false as an explanation of the observed g: **whatever the trigger does, the
stock configuration already extracts the most turn performance of the three.**

What the historical "4.7–5.8 g sustained against a lift limit near 11" actually was: **the worst
corner of the envelope.** Sustained g on the stock jet is strongly altitude- and speed-dependent —
4.72 g at 15 kft/180 m/s but **7.84 g at 5 kft/260 m/s**, with instantaneous peaks to **10.76 g**.
A2 independently measured the in-plane tracker sustaining nz p50 6.45 / p90 8.94 / max 12.83 in a
live engagement. The airframe is not the limiter; **what the policy commands** is.

Two numbers worth carrying forward, both new:

- **Sustained turn radius is set by ALTITUDE, not entry speed** — ≈795 m at 5 kft, 860 m at 10 kft,
  ≈940 m at 15 kft, essentially flat across 180–260 m/s entry (the jet converges to the same
  sustained condition). §5's symmetry arithmetic used R ≈ 980 m ⇒ 600 m of gun range is ~36° of arc;
  at the measured **795 m** it is ~**44°**. Fighting low materially widens the gun window.
- That pulls against the already-executed disproof "block at 5000 ft → 6/6 hard decks at 15.5–17.9 s"
  (§3). Low altitude buys turn geometry and costs vertical room. Recorded as a tension, not resolved.

⚠ **`.jsbsim-variants/` is a measurement fixture, not a shipped change.** Nothing about the sim's
aircraft changed; the stock root is untouched and the variants are gitignored and regenerable. Per
§6's "deliberately deferred", shipping an FCS change was always a separate user decision informed by
this number — and the number says **do not ship one**.

**Consequences.** D-A's three defects are all real in source and none of them is a lever — the
sticks were never going to teach BFM even fully fixed, because they do not fly it. **C1's gate
("only if A2 shows the corrected stick is worth copying") returns NO** and C1 is closed as specified;
there is nothing to copy. No behavioural change was shipped to `train_parallel.py` from A2: spending
a collector-behaviour change on a measured non-lever would be churn in a mid-experiment file, and the
handoff's own rule is one edit per run. The live lever is **B1**, which A2 independently reinforced.

### B1 — the reward's WEZ is now the real WEZ (2026-09-05) — **edit shipped, run pending**

**What shipped.** Four source sites, one lever:

1. `constants.py` — `PHIT_RANGE_M` / `PHIT_PER_ROUND`, the Monte-Carlo measured per-round hit
   probability at 200/300/395/440/500/600/700/900/1200 m =
   **1.000 / 0.983 / 0.834 / 0.701 / 0.484 / 0.202 / 0.060 / 0.002 / 0.000**; plus
   `LETHAL_RANGE_M = 600.0`.
2. `geometry.py` — new `p_hit_range(rng)` (interp, clamp 1.0 low / 0.0 high) next to `sigma_track`;
   `range_factor = p_hit_range(rng)` replaces `clip((R_EFF − rng)/R_EFF)`.
3. `reward.py` — the per-round WEZ bonus gate gains `and rg.rng <= C.LETHAL_RANGE_M`.
4. `bfm_detect.py` — its own `LETHAL_RANGE_M = 600.0` becomes a **re-export** of the constant, so the
   reward's `in_solution` and the D5 `lethal` tick can never disagree about what a killing shot is.

**Where the curve came from.** `controllers/rl/_diag_phit_range.py`, whose NumPy integrator is pinned
tick-exactly to the sim's `ballistics.rs` by a new cross-language golden trajectory
(`crates/astrum-sim/tests/ballistics.rs::golden_trajectory_pins_the_python_mirror` ⟷
`test_gunlay.py::test_matches_the_rust_golden_trajectory`; Rust suite 7/7 green). The sim's gun has
**no closed-form p_hit** — 8 mrad cone, 1030 m/s muzzle, rounds are entities stepped with
drag/gravity/Coriolis and resolved by segment-point distance — so measurement was the only option.

Conditions the table hardcodes, stated because it is now a reward constant: 5000 m, shooter 230 /
bandit 200 m/s, pure tail chase, bandit at **5 g**, and the **COMPUTABLE** lay (constant-velocity
`gunlay.intercept` — what an FCS can actually solve). The decay is **target maneuver during time of
flight, measured not assumed**: with the CLAIRVOYANT lay the same 5 g case still reads 0.675 at
600 m vs 0.202 computable. g-sensitivity is ~10× (1 g reads 0.594 at 700 m), so against a
non-maneuvering target the curve under-credits — accepted, because the T3 rides its AoA limit in the
break and there is no non-maneuvering opponent in this curriculum.

**Measured effect of the shipped edit** (re-run against the real tree, not the sandbox):

| rng | old `range_factor` | new | | K_AIM leg (perfect lay) | old | new |
|---|---|---|---|---|---|---|
| 150 m | 0.875 | **1.000** | | 1200 → 600 m | 30.00 | **12.12** |
| 400 m | 0.667 | **0.819** | | 800 → 400 m | 20.00 | **47.29** |
| 600 m | 0.500 | **0.202** | | 600 → 150 m | 22.50 | **47.88** |
| 700 m | 0.417 | **0.060** | | 200 → 150 m | 2.50 | **0.00** |
| 900 m | 0.250 | **0.002** | | *episode total* | 52.50 | **60.00** |

10 rounds on a **perfect** lay, static co-speed geometry: 400 m **+0.53**, 600 m **+0.51**,
700 m **−0.19**, 1100 m **−0.19** (and the last two now feed the waste ramp, `streak → 10`).
Previously the 1100 m case paid **+0.51** — the reward was paying full "this round connects" credit at
a measured p_hit of **0.000**, which is precisely the standoff spray the policy had learned.

**Three things the de-risking pass caught that the original design got wrong.** Worth recording,
because each would have cost a run:

- **This is not a tightening, it is a REALLOCATION.** Inside ~460 m the new factor is *higher*
  (400 m: 0.667 → 0.819), so `K_AIM = 60` — the largest coefficient in `constants.py`, 2000× `K_SOLUTION` —
  pays *more* in the c87 kill window. The episode total is telescoping-bounded (52.5 → 60.0, **1.14×**,
  far under the recorded 4–20× shock bar); what moves is *where* it is earned: out of the 1200→600 m
  standoff leg and into the 600→150 m conversion leg. That is the intended lever, now quantified.
- **The sub-200 m flat spot is real and is kept deliberately.** p_hit saturates at 1.0, so closing
  200 → 150 m pays **2.50 → 0.00**. Not patched with an invented slope: the *angular* gradient is
  untouched, `K_CLOSE` is already clipped at `R_CLOSE_MIN` = 250 m, and c93 found range is not the
  chase limiter. Flagged in `constants.py` so it is a decision, not a surprise.
- **`R_EFF` must stay 1200 m.** `in_range` is its only live reader and the **scripted T3's own
  trigger fires on it** (`opponent.py:327`). Collapsing `R_EFF` onto 600 m — the tempting cleanup, and
  one `bfm_detect.py`'s old comment actively invited — would stop the benchmark shooting in the
  600–1200 m band and void every score measured against it: a [[t3-break-ungated-pull]]-class silent
  benchmark change. Verified: with `R_EFF = 600` the env suite fails 6 tests today but only **3** after
  B1, i.e. B1 destroys three *incidental* R_EFF guards. Replaced with an explicit one.

**Tests.** `env-test` **91 passed** (88 + 3 new); `rl-test` **136 passed / 0 failed**; Rust
`ballistics` 7/7. The three new tests are the ones the suite structurally could not have caught —
the pre-existing suite passes B1 **unchanged, 88/88**, because every assertion re-derives its
expectation from the same `gun_quality` it is testing:

- `test_p_hit_range_is_the_measured_curve` — knots, clamps, monotonicity, and both *directions* of
  the reallocation (materially lower past 700 m, higher at 400 m) so a silent re-fit of a
  `K_AIM`-multiplied curve is loud.
- `test_lethal_range_gates_the_wez_bonus` — 600 m **inclusive** pays and forgives the burst; 700 m
  and 1100 m perfect lays are waste. (`<=` not `<` matters: two existing tests sit at *exactly* 600 m
  and `<` fails them.)
- `test_r_eff_is_the_opponent_benchmark_not_the_lethal_envelope` — pins `R_EFF == 1200`, that the T3
  still fires at 700/900/1100 m and holds at 1300 m, and that `bfm_detect.LETHAL_RANGE_M` **is** the
  constant. This is the invariant that keeps c94/c95 and everything after comparable.

**Two measurement-integrity sites fixed, not just noted** (the c92 "a re-scaled window fakes a
result" lesson):

- `controllers/rl/_verdict_naivelay_tmp.py:47` held a hardcoded *copy* of the old linear ramp. That
  shim exists to change exactly one thing — the lay — so post-B1 it would have differed from
  production in **two** ways and silently confounded the c86 re-measurement it exists to make. Now
  reads `rg.range_factor`.
- `_diag_t3_spray.py`'s `q = 0.05` default now means "inside ~734 m", not "inside ~1140 m", and at the
  700 m case it is usually run at, `q = 0.060` — a **0.010** margin above its own trigger.
  `train_parallel.py` cites that probe's "@700 m, 36–42 HP" as the damage-gate rationale, so a re-run
  is **not** comparable to the quoted numbers. Documented at the call site. Threshold shifts on a
  perfect lay, for the record: `q ≥ 0.60` 480 → 467 m, `q ≥ 0.30` 840 → **565** m, `q ≥ 0.05`
  1140 → **734** m — the last two move `bfm.py`'s `COACH_FIRE_Q` and the `wez%` verdict statistic.

**Not touched, deliberately.** `R_EFF`, `in_range`, obs idx 27, `TRACK_BUDGET` (B1 removes most of the
standoff-farm risk that `constants.py:280` cites as its justification, but that is a *second* edit and
the rule is one per run), and the five other diagnostics sharing the `q = 0.05` default.
`geometry.potential()` needed no thought: grep confirms it has **no production caller** — tests only —
so B1 perturbs no live PBRS through Φ.

**Obs note.** `range_factor` reaches the observation through `gun_quality` at **idx 29** only, so B1 is
a reward change **and** a one-column representation change — the c86 precedent. Pooled self-play
opponents read the same vector, so their idx 29 shifts too.

---

### B1 — the run is LAUNCHED (2026-09-05)

`runs/curriculum-96/`, warm from **c95** (the best base since the re-baseline), on c95's launch line
**verbatim** so B1's env edit is the single delta:

```
--ladder energy-merge --resume runs/curriculum-95/run.pt --start-stage 0 --steps 5000000
--save runs/curriculum-96/run.pt --save-every 50000 --workers 6 --rehearse-workers 2
--max-time-s 180 --merge-lateral-m 1400 --me-dalt-ft 2000 --stage-backstop-steps 5000000
--outcome-reward --merge-override-mode leadturn --merge-override-p0 0.75
--merge-override-steps 3000000 --merge-override-range-m 3000 --merge-override-ata-deg 25 --device auto
```

Header verified against c95's, line for line: `device cuda | workers 6 | start_stage 0 (RESUMED) |
obs 41 act 5 | collect prior-only | discount 0.9950 | horizon 3 | reward OUTCOME | seed_steps 5000 |
episode_length 4320 | steps 5,000,000`. `reward OUTCOME` is the `--outcome-reward` confirmation
([[outcome-reward-invisible-switch]] — not restored by `--resume`, not in the ckpt). Pre-launch
`pgrep -af train_parallel` was clean and the GPU idle at 15 MiB, per c95's double-trainer ops note.
Throughput ~1350 sps ⇒ ~62 min for 5M.

**Results: pending.** Judge per Phase B: `--aspects 0,30,45,55,60,75 --episodes 20 --jitter 0.5
--modes greedy`, plus the D5 ratio via `_bfm_verdict_tmp.py`, plus the head-on cell.

---

### D4 measured the RATE COMMAND, not commitment — instrument fixed (2026-09-05)

Found while building `controllers/rl/_bfm_verdict_tmp.py` (below), before it judged anything. **This
is an A1 spec defect, not a run result**, and it is the same class of error the A1 docstring already
warns about: two prior analyses of the same data disagreed about roll commitment purely from
statistic choice. Both were also wrong about the **quantity**.

`Control::Raw` writes `a[0]` to `fcs/aileron-cmd-norm` (`net.rs:50`). `fcs/fbw-override` has initial
value 0 and **nothing in this repo ever sets it** (grepped: no hits in `crates/astrum-sim/src`,
`crates/astrum-env`, `controllers/rl`, or the Python env), so `f16.xml:411` takes its *default* branch
and `aileron-cmd-norm` feeds the roll-rate PID at `f16.xml:364-396` (the model ships in the CONTAINER at
`/opt/jsbsim/share/jsbsim/aircraft/f16/f16.xml`; the repo copy readable on the host is A3's
generated `.jsbsim-variants/aircraft/f16/f16.xml`, which is gitignored and regenerable via
`make jsbsim-variants`)
(`roll-trim-error = aileron-cmd-norm − roll-rate-norm`, kp 3.0). `a[0]` is therefore a **normalized
roll-rate command** — confirming `[[f16-raw-is-flcs-nrt]]`, and corroborated by A3's own
`_diag_turnperf.py`, which steers it as `a = 2.0·(bank − φ)` for exactly that reason.

**Why that breaks D4.** On a rate interface a displacement roll is a *pulse* to establish bank, then
`a[0] ≈ 0` to hold it. Holding `|a[0]| > 0.5` for 1.5 s commands a sustained high roll rate — a jet
spinning about its velocity vector. So the old D4 was not merely noisy but **anti-informative**:
maximizing it means departing the manoeuvre. Its c95 reading of **0.17 s** would have been written up
as "the policy cannot commit to a roll" when it is approximately what correct rate-command flying
looks like.

**Shipped** (`bfm_detect.py`, `commitment()`): `longest_bank_hold_s` (`sign(φ)` constant ∧ `|φ| >
BANK_COMMIT_DEG = 45°`) and `longest_chi_hold_s` (`sign(χ)` constant ∧ `|χ| > CHI_ROOM_DEG = 45°`,
**valid ticks only** — an undefined-χ tick breaks the run rather than extending it or reading as an
in-plane pull). `Sample` gains `phi`, deliberately **required, not defaulted**: a silent `phi = 0.0`
would make any future trace-builder that forgets it report "never banked", a false negative on the
headline statistic. The rate-command run survives as `longest_rate_cmd_run_s`, relabelled a DITHER
measure where low is correct. `format_summary` now prints both holds, peak bank, and the rate-command
run tagged `= DITHER, low is correct`.

**Blast radius checked, and it is small.** `grep` found **no consumer** of `longest_roll_s` /
`commitment()` outside `bfm_detect.py` and its own `format_summary` — it never reached the collector
stat line the spec asked for, and no reward or gate reads it.

**What does NOT change.** D1/D2/D3 never read `ail`; they are predicates over `χ`, `nz`, `γ`,
`closure`, `aspect`, `cpa_miss`. So **A2's closure of C1 stands** — its conclusion rested on
D1/D2/D3 = 0.00 and on `|χ|` medians, not on D4. One mislabel to read past in A2's write-up: "`|β|` on
**committed-roll** ticks is p50 1.8–2.6°" means *high-rate-command* ticks, which if anything is the
harder test of the turn coordinator, so that refutation also stands.

**The `nz` scare, checked and clean.** Every one of D1/D2/D3 gates on `nz > 3.0`, so an inverted sign
would have made all three structurally incapable of firing — and would have made A2's "hollow stick"
an instrument artifact, since its sensitivity sweep only went down to `nz > 0.5`, still positive.
`geometry.py:81` sets `AState.nz = float(_finite(o.nz_g))` — g units, positive-up, and `geometry.py:399`
normalizes it as `(nz − 1.0)/NZ_SCALE` so 1 g level flight is 0. Valid. The "n_z sign inverted" note in
`[[f16-raw-is-flcs-nrt]]` is about the *elevator command* (`elev = −1` is max pull), not the reported
state.

**Tests:** `test_bfm_detect.py` gains `test_d4_measures_the_held_state_not_the_rate_command` — a
synthetic correctly-flown rate-interface roll (0.29 s pulse, 2.46 s hold at ~0 rate, roll-out pulse)
where the new holds read ≥ 2.46 s while the old statistic reads ≤ 0.29 s, plus the ±70° rocking trace
(both holds → 0.0) and the invalid-χ gap (χ hold breaks, bank hold unaffected). The old test is kept,
renamed `test_the_rate_command_dither_statistic_still_needs_magnitude_and_sign`, because its
sign-and-magnitude logic is still live — what changed is the conclusion drawn, not the number.
**92/92 env tests green.**

---

### The BFM detector, run on a POLICY for the first time (2026-09-05)

New harness `controllers/rl/_bfm_verdict_tmp.py`: loads a checkpoint, flies greedy (`eval_mode=True`)
episodes on `_verdict_tmp.py`'s env, records one `Sample` per **policy** step at `dt =
action_repeat/rate_hz` (24 Hz — matching A2's cadence exactly, so policy and scripted-stick numbers
are comparable), and prints `format_summary` next to the outcome columns. `--checkpoints` is a **list**
so D6 controls run in the same invocation against the same env build and seeds; the run goes first,
controls after. `--cell {chase,headon}` covers both the standard verdict geometry and the c94/c96
training geometry (reciprocal headings, sep 3000 m, lateral in metres). CPU by default so it never
contends with a training run.

**First read — c95, chase, aspect 30, greedy, 2 episodes** (thin, a plumbing smoke, but unambiguous):

| statistic | c95 |
|---|---|
| D1 displacement rolls / ep | **0.00** |
| D2 yo-yos / ep | **0.00** |
| D3 turning-room builds / ep | **0.00** |
| \|χ\| median | **12.2°** |
| lethal ticks | 38 |
| P(lethal \| baseline window) | 0.118 [102 windows] |
| D5 ratio | **undefined — there are no events to credit** |
| score | 41 HP dealt / 0 taken, 2 timeouts |

The best policy since the re-baseline flies **pure in-plane pursuit**: `|χ|` median 12.2° means the
lift vector sits essentially *on* the LOS for the whole engagement. It reaches the real gun envelope
(38 lethal ticks, 11.8% of baseline windows contain one) with **zero** detected BFM. So the same
D1 = D2 = 0.00 signature A2 measured on the scripted sticks now reproduces on a *learned* policy — and
the honest answer to "has anything in this program learned BFM" is **no, and it never needed to in
order to score**. D5 being undefined is the correct output, not a bug: a ratio needs events.

Do not over-read 2 episodes at one aspect. The full read (c96 first, c95 and c87 as D6 controls) is
part of B1's verdict.

### ⚠ The section below was REFUTED by its own D6 control within the hour — read both (2026-09-05)

Kept, not deleted, because the sequence is the lesson: a plausible, mechanism-consistent, table-backed
BFM finding was written up and then killed by the control the spec mandates. **c87 — older, no leadturn
prior, and strictly worse (5/5 decks) — scores HIGHER than c95 on every out-of-plane statistic:** D3
0.40/ep, mean bank hold **44.6 s** (vs 18.0), mean χ hold **13.0 s** (vs 6.26), `|χ|` median **37.0°**
(vs 13.5°). Per D6, any lift that also appears in a control is an artifact. It did, so it is.
Continue to "What the control actually exposed" below.

### ⚠ "The policy flies no BFM" is CELL-SPECIFIC — it does fly BFM on the head-on (2026-09-05)

Same checkpoint, same harness, same thresholds, the other cell — **c95, head-on, lateral 1400 m,
+2000 ft perch, 180 s, greedy, jitter 0.5, 5 episodes**:

| statistic | chase (asp 30) | **head-on (lat 1400)** |
|---|---|---|
| D1 displacement rolls / ep | 0.00 | 0.00 |
| D2 yo-yos / ep | 0.00 | 0.00 |
| **D3 turning-room builds / ep** | **0.00** | **0.60** |
| \|χ\| median | 12.2° | 13.5° |
| longest / mean **χ hold** >45° | — | **6.37 s / 6.26 s** |
| longest / mean **bank hold** >45° | — | **21.08 s / 17.98 s**, peak bank 180° |
| rate-cmd run (dither, low = correct) | — | 3.71 s |
| **lethal ticks** | 38 | **0** |
| score | 41 dealt / 0 taken | **0 dealt / 0 taken, 4 decks, 1 t/out** |

The blanket claim above is wrong as stated and is corrected here: **on the head-on c95 does displace
the lift vector off the LOS and hold it.** D3 fires 0.60/episode — and D3 is the strictest shape in the
family (`sign(ata_az)` constant ∧ `|χ| > 45°` for ≥0.5 s ∧ `cpa_miss` **increasing** during the hold and
**decreasing** after), i.e. it genuinely builds turning room and then uses it. A **mean** bank hold of
17.98 s means *every* episode contains a ~18 s sustained same-sign >45° bank: this is a committed,
sustained turn, which is exactly what D4 was rebuilt to be able to see. The chase-cell read stands on
its own terms; what it does not support is a claim about the policy in general.

Two cautions on reading the table:

- **`|χ|` median is a poor cell discriminator.** 12.2° vs 13.5° looks identical, yet one cell has zero
  out-of-plane events and the other has 0.60/ep and a 6.4 s hold. The median is dominated by the long
  quiescent stretches; the *holds* and the event counts are what carry the signal. Report holds.
- **The 3.71 s rate-cmd run and the 21 s bank hold are not the same moment** (both are per-episode
  maxima over 5 episodes). They cannot be: a continuous full-rate roll wraps `sign(φ)` at ±180°, so it
  caps a same-sign >45° bank hold at ~0.75 s (135° at ~180°/s). The long holds are sustained turns; the
  3.7 s command run is one episode's dither.

**The head-on wall is not only conversion geometry — the engagement is ALTITUDE-TERMINATED.** 4 of 5
episodes end in a deck with **zero** lethal ticks. This is *not* a harness artifact: §3 already records
the baseline head-on decking **4/6 at 53.7–75.0 s**, and A2's scripted sticks hard-deck in the merge too
(in-plane at 42 s, lowyoyo at 82 s, table in the A2 section). Decking is endemic to this merge geometry
across policies *and* hand-code. So c95 is buying its turning room with altitude it does not have, and
c94's diagnosis (the wall is the conversion, ata 124°, `alpha_req` 146°) is at best half the story: a
large share of these episodes never reach a conversion attempt at all because the jet is dead.

Note this cuts against the obvious fix: §3 records "stage the merge lower" as **dead by execution**
(block at 5000 ft → 6/6 hard decks at 15.5–17.9 s). Less vertical room is strictly worse, so the lever
is *not* more room to displace into — it is either an altitude/ground-proximity term the policy can
**see** and be scored on, or a shorter commitment. Do not act on this before B1's verdict; recorded here
because it is the first evidence that the head-on failure is partly a **survival** failure rather than
purely an aiming one, and 5 episodes is thin.

### What the control actually exposed: D4's holds had no load clause (2026-09-05)

The D6 control (c95 and c87, one invocation, head-on, lat 1400, 5 eps each):

| statistic | c95 (run) | c87 (**control**) |
|---|---|---|
| D3 turning-room / ep | 0.60 | 0.40 |
| mean bank hold >45° | 18.0 s | **44.6 s** |
| mean χ hold >45° | 6.26 s | **13.04 s** |
| \|χ\| median | 13.5° | **37.0°** |
| lethal ticks | 0 | 0 |
| decks | 4/5 | **5/5** |

A checkpoint that dies in *every* episode cannot be the one doing more BFM, so the statistics were
measuring something else. The bug: **D1, D2 and D3 all carry an `nz > NZ_LOADED` clause and the two D4
holds did not** (`bfm_detect.py:334`, `:420`, `:471` have it; the `_longest_hold` shipped in the D4
correction did not). NZ_LOADED exists precisely to make a detector "immune to the passivity failure
mode — a wings-level drift is not a manoeuvre", and the holds were the one place it was missing. c87's
44.6 s "commitment" is an unloaded wings-over descent.

This is the *second* D4 defect of the same species as the first: the original measured the wrong
QUANTITY (rate command vs held state), this one measured the right quantity without the qualifying
clause that makes it mean anything. Both would have been reported as findings.

**Fixed** (`bfm_detect.commitment()`): both holds now require `nz > NZ_LOADED`, and — because a *loaded*
hold can still be a high-g spiral into the ground — each hold now reports the altitude it spent,
`bank_hold_dalt_m` / `chi_hold_dalt_m`, summarized as signed **means** over episodes and printed on the
D4 lines. Duration alone prices a manoeuvre at zero; on this map altitude is the currency BFM is bought
with, and a 44.6 s hold that spends 4000 ft is a death spiral whatever its duration. `Sample` gains
`alt` (from `me.pos[2]`, read only for this). D1/D2/D3 unchanged. Tests: new
`test_d4_holds_require_load_and_report_the_altitude_they_spend` pins that a steep *unloaded* bank
(nz 1.2) yields hold 0.0 while the same geometry at nz 5.0 yields the full hold, that `dalt` is signed
and matches a known descent rate, that a level and a diving hold are **identical on duration and
separated only by dalt**, and that no-hold reports 0.0 rather than a stale number. **93/93 env tests.**

⚠ Method rule this earns: **the D6 control is not a formality to run after writing up a finding — it is
part of the measurement.** Both BFM claims made in this program before running one were wrong. Also:
`|χ|` median is a poor discriminator (12.2 vs 13.5 across cells differing by 0.60 events/ep), and a high
`|χ|` median can mean the *jet is dying*, not manoeuvring — c87's 37.0° is the spiral, not skill.

#### Re-run on the CORRECTED instrument — the ordering reverses (2026-09-05)

Same two checkpoints, same invocation, same seeds, load-gated holds + `dalt`:

| statistic | c95 (run) | c87 (control) |
|---|---|---|
| D3 turning-room / ep | **0.60** | 0.40 |
| bank hold mean (nz>3) | 13.37 s, **dalt −744 m** | 39.76 s, **dalt −2642 m** |
| χ hold mean (nz>3) | **5.40 s** | **2.45 s** (was 13.04 ungated — a **5× collapse**) |
| \|χ\| median | 13.5° | 37.0° |
| lethal ticks | 0 | 0 |
| decks | 4/5 | 5/5 |

The load clause collapsed c87's χ hold 5× and barely touched c95's (6.26 → 5.40), so **c95 now leads
c87 2.2× on the χ hold and 1.5× on D3**, instead of trailing 2×. The artifact is gone and a weak,
5-episode, same-direction signal is left. Do not promote that to a finding on 5 episodes.

c87's numbers are internally contradictory in a way that names the pathology: **39.76 s of loaded,
constant-sign bank but only 2.45 s of χ hold** means `sign(χ)` flips repeatedly *underneath* a steady
banked pull. The jet is turning hard while its plane of motion is **not held relative to the LOS** —
turning, but not manoeuvring against *him* — and it spends 2642 m doing it. c95's 5.40 s same-sign
loaded hold is LOS-referenced.

**The load-bearing new number is `dalt`: both checkpoints only know how to turn by spending altitude**
(c95 −744/−801 m per hold, c87 −2642 m), and everything decks. So the head-on is an **energy-management**
failure. Two things close the obvious responses:

- It is **not** representation blindness. `build_observation` already carries
  `(alt_agl − DECK_AGL)/C.ALT_REF` as feature index 1, and γ / sink rate are recoverable from
  `sin/cos theta` + `alpha` + airspeed — which §3's own R² 0.82–1.00 probe says an MLP reconstructs. An
  "add altitude-rate to obs" lever falls to the already-recorded refutation.
- The safe corner of the tradeoff was **already reached and was passive** (c85: "survival solved but
  PASSIVE"). c95/c87 sit in the aggressive corner and die. **Neither corner converts** — that is the
  structural statement, and it is why this is not a reward-weight tuning problem.

Also note the head-on produces **zero lethal ticks** for either checkpoint across 900 s of flight, where
the chase produces 38 in 120 s. The head-on is not near-converting; it yields no gun solution at all,
which bounds what any aim-side fix (B1 included) can show there.

### B1 VERDICT — 4 → 45 kills, the largest measured gain of the program (2026-09-06)

`runs/curriculum-96/run.pt`, `collected = 5,000,847` ✓. Greedy, out-of-band, `--aspects
0,30,45,55,60,75 --episodes 20 --jitter 0.5`. **No saved c95 verdict existed, so c95 was re-measured
in the same invocation format** rather than compared against a remembered number — this program has
already recorded one wrong-baseline error ([[curriculum-94-headon-lateral]]).

| aspect | c95 kills | c96 kills | c95 spent | c96 spent |
|---|---|---|---|---|
| 0 | 4 | **8** | 121 | 261 |
| 30 | **0** | **19** | 301 | **169** |
| 45 | **0** | **18** | 179 | **137** |
| 55 | 0 | 0 | 112 | 460 |
| 60 | 0 | 0 | 182 | 358 |
| 75 | 0 | 0 | 258 | 511 |
| **ALL** | **4 / 1 loss**, NET +24.6, 38% | **45 / 1 loss**, NET **+42.1**, **52%** | | |

**The mechanism is confirmed, not merely correlated.** At 30° and 45° c95 was already 95–100%
net-ahead and dealing 55.5 / 71.4 HP while converting **zero** kills — the recorded finish wall. c96
kills 37/40 there while firing FEWER rounds (301→169, 179→137). That is exactly B1's predicted
mechanism: the old `range_factor` paid linearly out to `R_EFF` = 1200 m where p_hit/round is 0.000, so
the policy sprayed from a median ~1519 m; paying on the sim's own measured p_hit curve made it close
first and shoot less. Fewer rounds + more kills is the signature that distinguishes this from "it just
got more aggressive".

**What B1 did NOT do: the ≥55° wall is completely intact** — 0 kills at 55/60/75, all 60 episodes
timing out, with no deaths and no decks. Note the *direction* of the change there: c96 spends MORE
rounds than c95 (460/358/511 vs 112/182/258) and deals 11.4/8.5 vs 0.4/0.0. So at high aspect the
honest WEZ made it engage harder and it still cannot finish. The trigger was never the binding
constraint at high aspect, which retires the last aim-side hypothesis for that wall and leaves the
§10 diagnosis as the standing explanation.

Caveat on scope: this is the CHASE benchmark; c96 trained on the `energy-merge` head-on ladder. The
head-on cell produced zero lethal ticks for c95 and c87, so B1 had nothing to convert there.

### B1's second half: the D5 ratio, with c95 and c87 as D6 controls (2026-09-06)

The handoff's judging rule for B1 was "the standard greedy verdict **plus** the A1 detector's D5
ratio". Same invocation, `--cell chase --aspects 30,55 --episodes 6 --jitter 0.5`, c96 first and the
two controls after, on the load-clause-corrected instrument.

| | c96 asp30 | c96 asp55 | c95 asp30 | c95 asp55 | c87 asp30 | c87 asp55 |
|---|---|---|---|---|---|---|
| kills / 6 | **6** | 0 | 0 | 0 | 0 | 0 |
| lethal ticks | **166** | 13 | 138 | **0** | 19 | 27 |
| D1 /ep | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.17 |
| D2 /ep | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| D3 /ep | 0.00 | 1.50 | 0.00 | **2.50** | 0.17 | 0.00 |
| bank hold (mean) | **1.28 s** | 19.6 s | **55.6 s** | 6.3 s | 13.6 s | 6.4 s |
| bank `dalt` | **+12 m** | −479 m | **−1340 m** | −401 m | −210 m | −219 m |
| chi hold (mean) | 0.60 s | 3.21 s | 0.57 s | 4.02 s | 2.04 s | 2.93 s |
| D5 event hits | 0 / **0 w** | **4 / 9 w** | 0 / 0 w | **0 / 18 w** | 0 / 1 w | 1 / 1 w |
| D5 base rate | — / 0 w | **0.000 / 225 w** | 0.118 / 306 w | 0.000 / 161 w | 0.041 / 291 w | 0.093 / 291 w |
| D5 ratio | nan | **inf** | nan | nan | 0.00 | 10.78 (n=1) |

**1. B1 is confirmed to be an aim fix, not a BFM result — and this is the cleanest statement of it in
the record.** c96 kills 6/6 at aspect 30 with **D1 = D2 = D3 = 0.00**: not one displacement roll, not
one yo-yo, not one turning-room build. The program's largest gain (4 → 45 kills) was bought with
**zero** additional out-of-plane maneuvering. Nothing about B1 touched the thing the user actually
asked for. It also re-confirms, on the corrected instrument, the standing chase finding: scoring there
never required BFM.

**2. Event COUNTS do not discriminate, exactly as D5 was designed to expose.** On the asp55 wall cell,
**c95 produces MORE events than c96** — D1 0.50 vs 0.00, D3 2.50 vs 1.50, 18 event windows vs 9 — and
converts **none of them**: c95 asp55 has **zero lethal ticks in the whole cell**. A run that rolled
out of plane more often and did less with it is precisely the artifact the ratio exists to catch. Any
future claim reported as "D1/D3 per episode went up" is therefore not evidence of anything.

**3. ⚠ RETRACTED 2026-09-06 as a RANGE ARTIFACT — see D4 in §9.** This paragraph claimed the program's
first non-degenerate D5 lift. D4's second verifier stratified the same windows by whether their 8 s
credit interval ever reaches inside `LETHAL_RANGE_M` = 600 m, and the contrast collapses: the lift is
**4-of-5 among windows that are range-capable and 0-of-4 among windows that physically cannot score**,
and the 225 event-free windows are overwhelmingly range-incapable. `lethal` requires `rng ≤ 600 m`
(`bfm_detect.py:253`, `constants.py:106`), so a window whose credit-interval minimum range is 688-722 m
had `D5hit = n` **whatever the learner did with its lift vector**. What the data actually say is
"lethal ticks occur where the range is lethal", which is a tautology, not an event-conditional lift.
It also does not survive a seed change: at seed 7 the same cell reads P(lethal|event) 0.286 [7 windows]
with lethalTicks 13 → 5 and dealt 15.0 → 3.7. **Do not cite this paragraph as evidence for anything.**
The original text and its stated limits are kept below, unedited, because three later boxes cite it.

4 of 9
turning-room windows are followed by a lethal tick within 8 s; **0 of 225 event-free windows contain
one**. That within-run contrast is the strongest control the instrument offers, and it is the first
time any checkpoint has shown a non-degenerate D5 lift. Honest limits, stated because this program has
twice written up a BFM finding that its own control then killed:
- The ratio is `inf` because the base rate is *exactly* 0.000, so the effect size is unbounded below
  by this data — it says "no lethal tick at aspect 55 ever occurs outside a turning-room window", not
  "turning room multiplies lethality by N".
- 9 windows across 6 episodes. Same-direction, small-n, one cell.
- **c87 asp55 nominally shows a lift too (ratio 10.78)** — on a single event window with a single hit.
  I am reporting it rather than burying it, per the D6 rule that a lift appearing in a control is an
  artifact until shown otherwise. It is uninterpretable at n=1, and the discriminating control is c95,
  which has the most events of any checkpoint and zero lethal ticks.

**4. The `dalt` pricing paid for itself immediately.** c95 at aspect 30 sits in a **55.6 s mean loaded
bank hold that spends 1340 m of altitude** and kills nothing; c96 kills 6/6 with a **1.28 s** hold at
**+12 m**. Duration alone would have graded c95 as 43× more committed. ⚠ Confound to state: c96's
asp30 episodes end in a kill in **under 8 s** — that is why its D5 has *zero base windows*, an 8 s
event-free window cannot fit — so its short holds are partly an artifact of a short episode. The `dalt`
sign is not: c96 gains altitude across its hold, c95 trades away a kilometre and a third. This is the
c87 "~5 s finish" mode ([[curriculum-93-chase-autopsy]]) reappearing, now with an honest WEZ behind it.

Not reported as BFM evidence, per the detector spec: c96 asp30's `|chi|` median of 120.2° is the merge
geometry of a 6-second head-on-ish pass with a 0.60 s chi hold underneath it, not maneuvering — the
same trap c87's 37.0° median set earlier in this section.

---

### D2a — the actor's own σ, read on the executed path (2026-09-06)

`controllers/rl/_diag_actor_std.py`, c96 as RUN with c95 as the D6 control **in the same invocation**,
`--cell chase --aspects 30,55 --episodes 4 --jitter 0.5 --opening-steps 138`. Rolled **greedy** so the
state distribution matches every other measurement in the program; at each policy step the actor head
is additionally evaluated to draw what collection *would* have executed there. Three layers, because
they differ by more than an order of magnitude and the record has already quoted one while reasoning
about another: `sigma_pre` = `log_std.exp()` (pre-squash), `dev_post` = `|pi−mu|` in action units, and
`win36` = the 1.5 s (36-step) rolling mean of `dev_post`.

Confirmed first that `sigma_pre` is the real sampling scale: `world_model.py:86-90` is
`pi = mu + eps*log_std.exp()` with `log_std = fn.clamp_log_std(...)`, and `clamp_log_std`
(`functional.py:79-81`) is a **smooth tanh map** into `[low, high]`, not a hard clamp. With
`log_std_min/max = −10/2` that is σ ∈ [4.5e-5, 7.39], so the measured 0.43–0.90 is a freely-learned
mid-range value, not pinned at a bound.

**The ail (roll) channel — the axis χ comes from:**

| view | σ_pre p50 | \|mu\| p50 | rail (\|mu\|>0.9) | \|dev_post\| p50 / p90 | win36 p50 | **N_eff** | P(win36>0.5) |
|---|---|---|---|---|---|---|---|
| c96 asp30 full | 0.680 | 0.913 | 0.523 | 0.060 / 0.625 | 0.029 | 4.5 | **0.000** |
| c96 asp55 full | 0.780 | 0.170 | 0.184 | 0.302 / 0.816 | 0.065 | **21.7** | **0.000** |
| c95 asp30 full | 0.902 | 0.252 | 0.059 | 0.404 / 0.909 | 0.063 | **41.2** | **0.000** |
| c95 asp55 full | 0.755 | 0.172 | 0.115 | 0.357 / 0.829 | 0.061 | **33.9** | **0.000** |

`N_eff = (dev_post p50 / win36 p50)²` — the measured count of independent samples in the window. White
noise predicts exactly **36**. **P(|win36| > 0.5) = 0.000 in all 40 channel-views** (5 channels × 2
cells × 2 checkpoints × {full, open138}); P(>0.25) tops out at 0.037.

**1. Amplitude is not the limiter, and the actor-std-floor sub-lever is CLOSED.** σ_ail is 0.67–0.90 —
**4× the `σ_ail = 0.21` in the record**, and 13–18× above `min_std = 0.05`, so the missing floor could
never have bound even if it were wired to the pi path. Per-step `|dev_post|` reaches p90 **0.61–0.91 on
a ±1 channel**: single-step exploration is already near-maximal. The D2a pre-registration's
"σ_ail above ~0.5 ⇒ the coherence framing is wrong" branch therefore **fires** — and I am **overriding
my own pre-registration**, because the threshold was set on the wrong statistic. It assumed large
`sigma_pre` ⇒ large window mean; the data shows large `sigma_pre` *and* a tiny window mean
simultaneously, which the third pre-registered layer (`win36`, described in the script as "the statistic
the P≈1e-50 claim is about") measures directly. Recording the override explicitly rather than quietly
picking the reading I preferred.

**2. Coherence is the limiter, and it is now measured rather than argued.** N_eff = 41.2 / 33.9 on the
two well-sampled c95 cells against a white-noise prediction of 36.0 — statistically indistinguishable
from i.i.d. The record's "a 1.5 s coordinated deviation is a ~15σ event, P ≈ 1e-50 at σ_ail = 0.21"
reached the right conclusion through wrong numbers; the correct statement is **amplitude-independent**,
because N_eff is a property of the correlation, not the scale. Raising σ cannot help — it is already
saturating the tanh.

**3. ⇒ D2b's required parameter is now pinned by measurement, not guessed.** For AR(1) with
`a = exp(−Δt/τ)` at Δt = 41.7 ms and N = 36, the window-mean reduction factor gives N_eff = 36 (white),
7.9 (τ=0.1 s), 3.6 (0.25), 2.2 (0.50), 1.8 (0.75), 1.6 (1.0), 1.2 (3.0). To move N_eff from 36 to ≲2
requires **τ ≈ 0.5–0.75 s**, i.e. a ≈ 0.92–0.95 per control step.

**4. A hard design constraint on D2b that nobody had: the noise must be injected PRE-squash, and on a
railed channel post-squash noise is a no-op.** `dev_post ≈ σ·(1−mu²)` locally. Worked example, c95
asp55 open138 `elev`: `|mu|` = 0.967 ⇒ (1−mu²) = 0.065, σ = 0.659 ⇒ predicted median |dev| =
0.674·0.043 = 0.029 against **measured 0.025**. So wherever the mean sits at the rail the deviation
collapses regardless of σ. D2b's box already says "injection must be pre-squash"; this is the
measurement that makes it mandatory rather than tidy.

**5. ⚠ REFUTED BY ITS OWN CONTROL — my in-flight hypothesis that the kill cell is the one cell that
rails the roll axis.** The `full` view is seductive: c96 asp30 (6/6 kills) has ail `|mu|` = 0.913 /
rail 0.523 while every 0-kill cell sits at `|mu|` 0.17–0.25 / rail 0.06–0.18. But c96's asp30 episodes
are **135/139/151/128 steps ≈ 5.75 s** (they end in a kill) against 1440 steps = 60 s for every other
cell, so that sample *is* the opening. Phase-matched to the same 138 steps, ail rail is **c96 asp30
0.512, c95 asp30 0.533, c96 asp55 0.732, c95 asp55 0.446** — every checkpoint rails roll in the
opening, and the matched pair is a dead heat. The entire contrast was episode-phase artifact; the
`full`-view 0.17–0.25 is late-engagement drift. This is why `--opening-steps` exists, and it is the
third time this program has had a metric fake a difference through a window choice
([[curriculum-92-lateral-separation]], the D4 load clause).

**6. `elev` rail does not track kills either, phase-matched.** asp30: c96 0.223 vs c95 0.165; asp55:
c96 0.375 vs c95 0.601 — inconsistent in sign. The saturation *mechanism* (finding 4) is confirmed; its
*incidence* is cell- and phase-dependent and carries no outcome signal. The eye-catching `elev` rail
0.795 on c96 asp55 is a **late**-engagement number (0.375 in the opening) and must not be quoted bare.

Not claimed: this measures the amplitude an i.i.d. sampler offers *at states the greedy policy visits*.
Rolling stochastically would visit a different state distribution and the deviations would compound
through the dynamics instead of being independent per-step draws. It is not "the exploration the
collector achieves", and D2c is still the box that measures the flown action.

---

### D2b — the coloured-noise seam is landed, and it changed the recommended arm (2026-09-06)

Shipped as plumbing with **no verdict attached**, per the box. Four files:

- **`world_model.py:83-97`** — `pi(z, eps=None)`. `eps=None` reproduces `torch.randn_like(mu)`
  exactly, so the update path and every eval are untouched. Injection is **pre-squash**, which D2a
  made mandatory rather than tidy.
- **`agent.py`** — `COLOURED_CHANNELS = (0, 1)` (`ail`, `elev`: roll and g, the two axes that compose
  an out-of-plane maneuver) and `TDMPC2._explore_eps()`. `act()` now passes
  `eps=None if eval_mode else self._explore_eps()`. For `white` the helper returns the reference draw
  directly — one `randn` of exactly `A` values — so the **default config is bit-identical**, which is
  asserted rather than asserted-by-eye.
- **`config.py:54-77`** — the measured N_eff table (below) written in beside the knobs.
- **`tests/test_agent.py`** — five tests; `141 passed` on the full RL suite.

**The defect this fixes, restated concretely:** before this seam the run header could record
`explore_noise='pink'` while collection sampled white, because the sampler was only ever consulted
inside the dead `plan()`. The `ou`/`pink` arms were unreachable configuration.

**⚠ The measurement changed the recommended launch arm, and my first threshold was wrong.** I initially
asserted on lag-12/36 autocorrelation of the flown action and `pink` failed at lag 36 (0.070). The
threshold was guessed, not measured — so I measured the sources, and the *statistic* turned out to be
the problem: a sample ACF over a few-hundred-sample block with the sample mean removed is biased low
for a long-memory process. On **N_eff over a 36-step window** — the statistic D2a used, and the one that
actually decides whether a coherent 1.5 s deviation is reachable — the sources read:

| source | N_eff(36) | | source | N_eff(36) |
|---|---|---|---|---|
| **white** | **40.63** | | `pink` β=1.0 | 2.20 |
| `ou` τ=12 | 2.27 | | `pink` β=1.5 | 1.25 |
| `ou` τ=18 | 1.86 | | `pink` β=2.0 | 1.07 |
| `ou` τ=36 | 1.41 | | `pink` β=2.5 | 1.05 |
| `ou` τ=72 | 1.23 | | | |

Three things follow. **(1) D2a's closed-form AR(1) table is confirmed to within 3%** — it predicted
2.2 / 1.8 / 1.4 for τ = 12 / 18 / 36 against measured 2.27 / 1.86 / 1.41. **(2) D2a's own N_eff numbers
are validated**, which matters because they carry the finding: this `WhiteNoise` source reads **40.63**
on the same estimator that read the actor's roll axis at **41.2 / 33.9**, so the actor's exploration is
quantitatively *identical to white noise* and its N_eff was not an estimator artifact. **(3) `ou` is the
recommended launch arm** — τ is an explicit correlation time, its marginal variance is exactly unit by
construction, and its closed form is now validated. `explore_beta = 1.0`, the pink-noise-paper default,
is the **weakest** coloured option here (2.20, barely inside the ≲2 target), so a null from a β=1 arm
would be a weak null. The defaults are left at `white` / τ=36 / β=1 — no default was churned — but the
table is in `config.py` so the launch picks with numbers.

Still no verdict claimed: this establishes that the lever is *reachable and correctly sized*, not that
it converts. That is a training run, and per the §6 amendment it must be judged **on the chase**, where
the reward is not confounded.

### D2c — the coherent deviation is REACHABLE closed-loop: 0.000 → 0.256 (2026-09-06)

`_diag_actor_std.py --arm exec`, added because every sanctioned probe hardcodes greedy. The `exec` arm
flies `eval_mode=False` — the real collection path — so deviations compound through the dynamics and the
state distribution shifts with them, which is exactly what D2a disclaimed. c96, chase, 4 eps/cell,
`--jitter 0.5`; white and `ou` τ=36 as paired arms.

**`ail` — the roll axis, i.e. χ:**

| arm | N_eff | P(win36>0.25) | **P(>0.5)** | P(>0.75) |
|---|---|---|---|---|
| asp55 white | 8.2 | 0.035 | **0.000** | 0.000 |
| asp55 **ou τ=36** | 1.2 | 0.486 | **0.256** | 0.102 |
| asp30 white | 12.4 | 0.019 | **0.000** | 0.000 |
| asp30 **ou τ=36** | 0.8 | 0.239 | **0.093** | 0.040 |

**1. The blocking quantity moved from never to routine.** P(a 1.5 s window holding a mean roll deviation
past half deflection) was **0.000 in all 40 channel-views** of D2a and is **0.256** here — and 0.102 even
past *three-quarter* deflection. This is the first time in the program's record that the exploration
process can propose a multi-second coordinated deviation at all. §10.1's claim that it "cannot propose a
maneuver" is now **false as of this commit**, which is the point of the commit.

**2. Pre-squash injection is doing visible work on the load axis.** `elev` asp30 reaches P(>0.25) = 0.325
and P(>0.5) = 0.076 in the `ou` arm despite sitting at the tanh rail — impossible for a post-squash
perturbation. D2a predicted this and D2b's placement delivered it.

**3. A caveat I nearly mis-read as a leak.** `rud` is **not** in `COLOURED_CHANNELS`, yet its exec-arm
N_eff fell 11.0 (white) → 3.0 (ou). That is not correlation leaking into its draw: `dev = a − mu` also
carries the slow drift of the squash gain `(1−mu²)` as the jet maneuvers, and the `ou` arm flies a more
dynamic trajectory, so an uncoloured channel's `dev` inherits low-frequency structure from `mu`. The
isolating test pins `mu` with a fixed observation and confirms channels **2/3/4 all hold N_eff > 15**
(`tests/test_agent.py::test_uncoloured_channels_stay_white`, extended to ch2 for exactly this reason).
**Do not read exec-arm N_eff on an uncoloured channel as a plumbing check** — it is confounded by design.

**4. Observation, explicitly NOT a claim.** c96 asp30 episode lengths were `[133, 138, 144, 122]` under
`ou` — all short, matching greedy, where short = the kill finishes — but `[133, 1440, 151, 1440]` under
white, i.e. white exploration broke two of four finishes while `ou` preserved all four. Consistent with
coherent noise being less destructive per unit of exploration than jitter, but n=4, episode length is a
**banned** kill proxy ([[curriculum-91-altitude-budget]]), and this was not scored by the detector.
It motivates the run; it is not evidence from it.

**What is now established and what is not.** Established: the lever is *reachable, correctly sized, and
lands on the intended axes closed-loop*. Not established: that it converts to BFM or to kills. That is a
training run, judged **on the chase** (unconfounded per the §6 amendment) with `bfm_detect` and its D6
controls, on `--outcome-reward`, at `explore_noise=ou --explore-tau-steps 36`.

### D0 — two single-location refutations, copied into the record (2026-09-06)

D0's remaining items. Each of these was recorded in exactly **one** place, and each has already caused
re-proposals of a closed idea (three, between them). They live here now.

**(a) The `_refute_noise_band.py` band reading — and why D2c does NOT contradict it.** The record's one
prior measurement of coloured noise on an *executed* action reported the 1.5 s (36-step) window-mean
aileron **std** at **0.360 white → 0.322 OU** — i.e. coherent noise moved the executed band *down*, and
the lever looked dead. That reading is **correct for what it measured and does not govern**, because
`_refute_noise_band.py:23-31` imports `roll`/`paired_step` from `_diag_explore_noise`, and
`paired_step:143-154` calls `_plan_explore` — a verbatim copy of `TDMPC2.plan()` (`:157-163`). So it
injected the arm's sample as **MPPI's final action draw**: on the dead path (`cfg.mpc = False` in every
shipped checkpoint), at MPPI's `std[0] ≈ 0.21`, *after* the elite mean and *after* the clamp — three
places downstream of the actor's own Gaussian. Two further differences make the numbers non-comparable:
it reports a **std of the absolute executed command** (which is dominated by the policy's own mean, not
by the perturbation — its own `noise share of var` column exists to say so), whereas D2c reports
`P(|window mean of the DEVIATION a − mu| > D)`; and the perturbation was **post-squash**, which D2a
measured to be a no-op on a railed channel (`dev_post ≈ σ·(1−mu²)`) and roll is railed 0.45–0.73 of the
time in every checkpoint and cell. **What governs:** the same family measured on the **live pi path,
pre-squash**, goes 0.000 → 0.256 (D2c above). Both results stand; they are different injection sites.
The lesson is the general one — **an exploration measurement is only about the path collection takes**,
and until D2a nobody had checked which path that was.

**(b) The c62 pure-outcome catastrophe** (until now only in the `reward.py:106-110` docstring).
reward-v19 (2026-08-31, the BFM pivot, superseded the same day) dropped the five dense pure-pursuit terms
entirely under `outcome_mode`. c62 **catastrophically collapsed the champion's fine-control finish —
18/24 decked at aspect 45/55° by only 1.15M steps**, against the identical verdict on the champion
baseline of **0 losses / 9 kills**. Mechanism: the dense tracking term *is* what maintains the ≤55° kill;
removing it everywhere — **including the rehearsal anchor** — left nothing to preserve fine control and
the warm policy decked. The fix that shipped is reward-v19b (`:112-122`): `outcome_mode` keeps full dense
pursuit and merely *adds* the outcome extras. ⇒ **"just go outcome-only" is tried, dated, and decked the
champion**; see also §10.6. Any future proposal that removes a dense term must state how the ≤55° finish
and the rehearsal anchor are preserved, and must exclude the anchor by stage (the `me_dalt_ft` /
`merge_alt_ft` exclusion path at `train_parallel.py:3236-3260`), not through the shared `env_kwargs`.

D0's third item (the §10.3 `reward.py:332` mis-cite) was applied directly to §10.2/§10.3 when this board
was authored — see the ⚠ banner at §10.3 — and went further than the board asked: the charge is **two**
terms, `energy` measured **positive** in all three arms, and `aspect` is sign-inconsistent.

---

### The free tier — D1a, D3a, D4, D5, D6a, all five run and all five adversarially verified (2026-09-06)

One orchestrated pass: five boxes measured in parallel, each then audited by **two independent
adversarial verifiers** that were instructed to attack the statistic, not the prose. 15 agents, 0 errors,
~83 minutes, 1.81M tokens. Every one of the ten verifiers wrote its own harness or re-ran the box's code
with the missing controls, and **eight of the ten found the reported statistic unsound**. The verdicts
below are the *corrected* ones, after the audits.

⚠ **Process defect in this pass, recorded so it is fixed next time:** the verifier prompts shipped with
their `${res.*}` template placeholders **uninterpolated**, so no verifier ever saw the reporting agent's
write-up. All ten located the artifacts on disk and graded the *box* bottom-up instead. That accidentally
made the audits stronger (ten genuinely independent reconstructions) but it means `verdict_upheld` in this
pass means "the evidence supports the box's pre-registered criterion", **not** "the prose was checked
verbatim". Interpolate the placeholders in any future verify stage.

| box | reported | **corrected** | what changed |
| --- | --- | --- | --- |
| **D1a** | FAIL | **FAIL** (2/2 upheld) | leg holds; the *chase* clause is shown to be uninformative and the *headon* separation is a threshold artifact |
| **D3a** | PASS | **PARTIAL** | the measured half stands; **the named band R\* = 271-353 m is in no artifact** and three of its headline sentences are falsified by its own code |
| **D4** | PASS | **FAIL — the mechanism is vacuous on the run cell** | the pre-registered FAIL branch fired and was not applied |
| **D5** | FAIL | **INCONCLUSIVE** (2/2) | the statistic is an algebraic identity; but it surfaced **two new, source-level, unnamed mechanisms** — the most actionable output of the whole pass |
| **D6a** | FAIL | **FAIL** (2/2) | verdict survives, **leg replaced**: it dies on arithmetic, not on the depth curve, and the box's own statistic is banned |

---

#### D1a — the detector-gated-waiver family is DEAD as an attitude-and-load predicate

`controllers/rl/_diag_d1a_gate.py`; 180 episodes, five arms × two cells × identical seeds, one
invocation. **Both pre-registered FAIL conditions fire on the chase, at all nine (χ, nz) threshold
settings** (χ 30/45/60 × nz 2/3/4): `inplane` 0.0248 ≥ `highyoyo` **0.0000** at 9/9, and c87 ≥
min(c95, c96) at 9/9 — c87 is the single **highest-clearing arm of all five** at 0.0875 (c95 0.0658,
c96 0.0622), outright highest at 6/9. On the headon the two mechanical tests do not fire (inplane 0.0024
< highyoyo 0.1240; c87 0.0649 < both policies) but the PASS conjunction the box actually requires —
"`inplane` **and** c87 both near zero **and** the out-of-plane arms materially above" — is not met either:
c87 clears **6.5%** of ticks, **27× the in-plane control**, while decking 10/12.

**Decisive, and in the currency D1b would actually have paid in.** D1b rebates at excursion *close*, i.e.
**per excursion**, not per tick. In that currency c87 leads the yo-yo arm on **both** cells: 9.42 vs 2.33
gated excursions/episode on the headon (**4.04×**) and 6.00 vs **0.00** on the chase. There is no
threshold at which the waiver pays the maneuver more than it pays the program's dominant failure mode.

**Both verifiers upheld FAIL and then made it worse for the lever, in two ways the write-up had missed:**
- **The chase FAIL clause carries no information** — `highyoyo` flies at all-tick nz p50 1.17-1.18, so the
  `nz > 3` clause zeroes it *arithmetically* (0 gated ticks in 27,315). That is the `D_straight` failure
  C-7 already banned as a control. The chase clause could not have come out the other way.
- **The headon "survival" is an egress, and a threshold artifact.** The only arm materially above the
  policies there is `highyoyo`, which is `left_bubble` **12/12** on the headon (24/24 on the chase) at
  nz p50 1.18, gated ticks at rng p50 **1590 m**, aspect p50 **159.3°**, and **+2045 m total altitude** —
  it climbs away. That is the documented c70 zoom-away signature, not a maneuver. Its headon gate also
  collapses 0.1240 → 0.0203 (6.1×) at nz>4. And the gate ranks the **deck-bound descending spiral** 2nd
  and 3rd: c95 headon gate 0.0946 with the longest holds of any arm (dur p50 2.58 s, max 9.58 s) and
  **−14,807 m** of altitude — exactly the killer D1b's own judging note named.

**Two instrument traps this box exposed, both of which outlive it:**
1. **Threshold tightening fragments holds, it does not purify them.** c96 headon at χ45, nz 3→4 drops
   GATEfrac 22% but raises excursions/ep 7.08 → **25.42** while collapsing duration p50 from 40 steps to
   **1 step**. Any future gate must report duration *and* rate, or a tighter bar will read as "more BFM".
2. **`chi` is ill-conditioned exactly where the policy arms' chase clearances live.** `out_of_plane_angle`
   returns `valid=False` only when `|los_perp| < sin(5°)` (`bfm_detect.py:168`), and since `los` is a unit
   vector `|los_perp| = sin(ata)` — so just outside the 5° cone χ is numerically meaningless while still
   reported valid. c96 chase/asp30's gated ticks sit at **ata p50 8.6° / p90 9.2°** against that 5° bar,
   with |χ| p50 156.1° and **16.5% of ticks outright invalid**: at ata 8.6° a 10 m transverse bandit
   displacement at 500 m moves |χ| by **7.62°**. c87's clearances are well-conditioned (3.9% invalid), so
   the comparison was biased *toward* the arms and the FAIL is if anything understated. **Raise
   `CHI_VALID_ATA_DEG` or report an ata-stratified χ; never compare χ across arms with different ata.**
3. Presentational, but it has already misled one reader: `f_chi` is normalised over **valid** ticks while
   `f_nz` / `f_out` / `f_gate` are over **all** ticks (`_diag_d1a_gate.py:373-377`), so the printed clause
   fractions cannot be multiplied back to the gate.

**Binding clauses, for the record:** `highyoyo` is zeroed by **nz** (0.0236; nz p50 1.18) — the load
clause rejects the entire deliberate out-of-plane arm, which is A2's already-closed result. `inplane` is
bound by **χ** (0.0522). c87 headon's `¬in_zone` is inert (0.9964). c95 chase's `¬in_zone` is 0.4967 — the
one arm that spends half the chase *inside* the reward's own convert zone.

**Altitude sign inverts between cells, so holds must always be priced in altitude.** Headon totals: c95
−14,807 m, c96 −4670, c87 −1830, inplane −260, highyoyo **+2045** (the only climbing arm is the hollow
stick). Chase totals: c87 **+3120**, c96 +647, c95 +608, inplane −2595. Outcomes: zero kills for every arm
on the headon; chase c96 12 kills / 12 timeouts, c87 1 kill / 3 decks / 20 timeouts. Headers verified —
c96 5,000,847 / c95 5,001,501 / c87 5,000,059.

⇒ **D1b is dead. Do not build the accumulator.** Every remaining escape route was checked: nine threshold
settings, both cells, both currencies (duty cycle and rate), and an independent re-run that reproduced
every pooled number to 4 dp. One residual worth banking: every gate fraction here is an **upper bound** on
what a waiver could pay, because `reward.py:390` makes the out-of-zone `K_LEAD` term one-sided
(`lead_r = raw_lead if (raw_lead <= 0.0 or in_zone) else 0.0`) — only ticks where `lead_error` *degrades*
are charged, so the payable set is (GATE ∧ lead degrading), a strict subset.

---

#### D3a — the dwell budget: the measured half stands, the NAMED BAND does not

`controllers/rl/_diag_dwell_budget.py` (973 lines) + `_dwell_budget_clean.txt`. Reported PASS. **Corrected
to PARTIAL**: one verifier INCONCLUSIVE, one upheld PASS while filing 13 defects. What survives is worth
having; what does not survive was the headline.

**Survives — measured, and independently re-verified from Rust:** gun constants match the Rust source
(`combat/components.rs:35-38` rate 100 rd/s, muzzle 1030 m/s, belt **511 rd**, dispersion 8 mrad;
`combat/resources.rs:27-29` damage 2, hit radius 8 m, ttl 3 s; `scenario.rs:53-54` hp 100). The
fractional-accumulator mechanism is real — `combat/systems.rs:63-73` **zeroes `gun.accum` on a released
trigger**, so trigger chatter destroys accumulated rounds. **Achievable dwell against the T3 is 0.00 s and
E[hits] = 0.00 in 16 of 16 arm × cell combinations, over 18,069 sub-600 m control steps = 752.9 s of
exposure with the trigger counterfactually held.** The learner's α cone is p99 **12.2-12.7°**. The T3's
break radius is 869-1204 m. No scripted arm ever entered 400 m on either chase cell.

**Does NOT survive, and the corrections are load-bearing:**
- **The named band R\* = 271-353 m appears in no A1 row of any artifact.** `ALPHA_CEIL_DEG = 16.7` is a
  hardcoded module constant with **no argparse flag** (`:132`), so no sensitivity sweep is possible and the
  only A1 table that exists prints **R\* = 403 / 362 / 365 / 499 m**. The 12.5° and 14.6° columns the
  headline band came from were never computed in an A1 row — the α follow-up ran `--parts a2,a2c`. The box
  requires the band to be NAMED; it is not named by any teed output.
- **"Inside R\* the dwell is unbounded by geometry, the T3 never relents" is falsified by a seed-matched
  re-run of D3a's own code.** `_d3a_verify_asp55.txt`, arm `lead_max` vs T3, chase-asp55: **29.0 s inside
  400 m** (362 steps in 200-300 m, 333 in 300-400 m). And the "never relents" claim is contradicted in
  D3a's own A2b table — chase-asp55/`lead_max` reads DEFENSIVE 25.2% / NEUTRAL 42.8% / OFFENSIVE 22.6%,
  and merge/`lagtrack_fixed` shows 3.8% ENERGY_RECOVERY. The 90.9-99.9% DEFENSIVE figure is true for
  chase-asp0/asp30 only.
- **"From ~450 m outward no stick input produces the gun solution at all" is falsified by the box's own
  instrument.** chase-asp30, T3, 500-600 m: **7.2%** of 405 control steps have `alpha_req ≤ 16.7°`
  (min 8.1°); chase-asp55, expert, 500-600 m: **21.0%** of 243 steps (min 3.0°).
- **"It is GEOMETRY, not aim" is unsupported on the cell it diagnoses.** All four scripted arms have
  independently broken aim (conv ≤ 6.6%), so they cannot separate the two. The discriminating datum is the
  reverse: expert vs the non-breaking dummy at 300-400 m has **96.6% of 592 steps inside the cone** at
  median `alpha_req` 9.3° — geometry fully available — and still converts nothing. **Aim is at least
  co-limiting.**
- **"A one-belt kill is arithmetically impossible past ~673 m" is a median dressed as an impossibility.**
  673 m is where *mean* rounds = 511; Binomial(511, 0.0978) gives **P(≥50 hits) = 0.52**. The
  effectively-impossible range is **~700 m** (p = 0.060 ⇒ P = 5e-4). Use 700 m.
- **`E[hits | trigger held]` has no magazine cap** (`:597` integrates 4.17 rd/step with no 511-round
  limit), so the control rows are physically unrealisable: `lagtrack_fixed` vs dummy at chase-asp0 prints
  **E[hits] 104.5/ep** = ~3456 rd/ep ≈ 7 belts, while actually dealing 10.3 HP.
- **`_coturn` treats n and α as independent** (`:788-816`): each row derives `r = V²/(g·√(n²−1))` from n
  *and* separately grants a 16.7° nose-off, but in a steady co-turn the nose offset **is** the α that
  produces the load, so (V, n, alt) fixes α. The row "V 230 / n 4.0 / 16.70° / r 1393 m" is not a
  physically consistent state.
- **⚠ D3a IS NOT REPRODUCIBLE, and its D6 control is not seed-matched.** `:648`
  `seed0 = args.seed + 1000*(hash(cell) % 7) + 37*opp_tier` — Python string hashing is randomised and
  `PYTHONHASHSEED` is unset in the container (verified: three invocations gave [0,2,1,4] / [3,4,2,3] /
  [4,1,5,2]), so the recorded command cannot reproduce the recorded numbers, and the `+ 37*opp_tier` term
  hands the Tier-1 control different seeds than the T3 arm. **Fix `hash()`-derived seeds before anyone
  re-runs this file.** A verifier's replacement harness (`_diag_d3a_verify.py`) imports
  `_diag_dwell_budget` read-only and seed-matches correctly — use that pattern.
- **The mandated D6 control set was omitted** (a tier-1 dummy was substituted, and run *second*). The
  record's own same-instrument measurement of c96/c95/c87 against this exact T3 at chase-asp30 is
  166 / 138 / 19 lethal ticks with c96 killing 6/6 — that is the control with power, and its existence
  makes "0.00 s achievable dwell" a statement about the **scripted arms**, not about the airframe.
- **`_dwell_budget_clean.txt` is missing its two most load-bearing tables** (three bare R3 headers with
  zero data rows at :65-67, an empty A2b table at :139-142): the JSBSim-spam filter took the data rows
  with it. Re-extract before citing.

⇒ **The honest reading of D3a is P2, not P1**: *no dwell budget was found at R ≤ 600 m under model
M = the four scripted trackers vs the fixed T3.* That bounds achievability **from below only** and — per
the box's own pre-registration — **motivates no task change**. It does not say the fight is unwinnable, and
the record's 166 lethal ticks for c96 says the airframe can do what the sticks could not. **D3b is
therefore not optional: it is now the only instrument that can answer the winnability question**, and it
must fix the four things above (α ceiling as a swept flag, magazine cap, α-n coupling, deterministic
seeds).

---

#### D4 — FAIL: the T3 nose-gate is real, but it is VACUOUS on the run cell

`controllers/rl/_diag_t3_release.py` + `_t3_release_out.txt`. Reported PASS. **Corrected to FAIL** — the
box's own pre-registered FAIL branch fired and was not applied. Verifier 1 upheld PASS on the letter of
"does ata ever exceed 65°"; verifier 2 wrote a new harness with the controls restored and showed the
pre-registration was about **loaded out-of-plane content**, not about any tick. Both verifiers
independently concluded the *causal* claim dies. The FAIL is the correct label.

**What is true and worth keeping — the nose-gate defect exists and is in none of the four record
locations.** `THREAT_CONE_EXIT_RAD` = 65° (`opponent.py:385`) lets the T3 leave DEFENSIVE on the
*learner's instantaneous pointing*. On c96 asp55, 163 of 1081 in-window control steps (15.1%) exceed 65°
with a window max of 97.8°, and 4 of the 9 turning-room windows each contain a DEFENSIVE→NEUTRAL release
at **rng 771-818 m** — far inside `THREAT_RANGE` (2500 m), so `threat_clear` can only have been satisfied
by the nose term — with the T3 at 225.8-228.0 m/s, far above `V_ENERGY_LOW` (105), so it is not the C-5
energy channel. Each release cancels a break committed for 11.9-15.2 s, so it is not dwell chatter. The
instrument itself is sound: both exact tick-aligned consistency checks passed in all six cells, 0
violations. **Record the nose-gate as a verified defect.**

**Why it is nonetheless vacuous as an explanation of c96's ceiling — five independent refutations:**
1. **The gate never fires on loaded out-of-plane ticks in the RUN, and fires constantly in the CONTROLS.**
   `P(ata > 65° | |χ| > 45 ∧ nz > 3)` = **0.0% (0/718)** on c96 asp55; **26.5%** (495/1866) c95 asp55;
   **34.6%** (297/858) c87 asp30; **66.6%** (721/1083) c87 asp55. Releases landing *on* a loaded
   out-of-plane tick: c96 asp55 **0 of 6**; c95 asp55 3/10; c87 asp30 3/6. The mechanism is a property of
   the checkpoints that *fail*, not of the one that scores.
2. **The base rate inverts the claim.** In-window ticks with ata > 65°: 163/1081 = 15.1%. Out-of-window:
   2561/7559 = **33.9%**. The nose gate is **2.2× LESS** likely to be satisfied while the learner is
   inside a turning-room window.
3. **"In a D1/D3 window" is window membership, not maneuver membership, and all four flagged releases are
   unloaded.** They occur at nz = 2.2 / 2.4 / 1.9 / 2.1 — **all below `NZ_LOADED` = 3.0** — at |χ| =
   139 / 124 / 167 / 172°, i.e. the lift vector past 90°, pointed *away*, not turning.
4. **The release-vs-`D5hit` anti-correlation is a pure range artifact** (and it is the same artifact that
   retracts B1's §9 point 3, above). The 4 windows containing a release have credit-interval minimum range
   **711 / 688 / 722 / 695 m**; `lethal` needs ≤ 600 m, so `D5hit = n` was geometrically impossible
   whatever the opponent did. The 4 `D5hit = Y` windows are the sub-600 m early ones (min rng 185-221 m),
   whose ata max is 44.9-48.8° — so `threat_clear` could not fire there either. The contingency table is
   **mechanically forced**, not evidence.
5. **Raw release counts are exposure-weighted and were compared across cells with 745-8640 near-ticks.**
   Per 1000 near ticks the run has the **lowest** asp55 release rate of the three checkpoints: c96 0.69,
   c95 1.27, c87 1.79. Any sentence built on "7 releases vs 13 vs 10" is reading exposure. Relatedly,
   c96 asp30's "0 releases / 0 windows" is an exposure artifact — it kills 6/6 at ~5.2 s so it has n=745
   steps against n=8640 for every other cell.

Two smaller items: the `D5hit` ledger was produced in a **second invocation that dropped c95 and c87**,
violating the same-invocation D6 rule (restored by the verifier: c95 asp55 has 18 lethal-capable windows,
a release in one of them, and **0 lethal ticks in the entire cell** ⇒ range capability is necessary and
not sufficient). And `_diag_t3_release.py:27` claims the `rg_t.ata` identity is "asserted numerically in
the header"; `:502` only prints it as prose. The identity holds by construction (the harness recomputes
`relative(learner, bandit)`, the same call as `opponent.py:480`), so this is a doc defect, not a validity
one.

⇒ **Stop here, exactly as the box instructed.** Do not build `Tier3CommitOpponent`. The nose-gate is
banked as a verified environment defect; it is not the thing standing between c96 and a high-aspect kill,
and fixing it would make the benchmark **easier** while explaining none of the ceiling.

---

#### D5 — INCONCLUSIVE as designed, but it found TWO new source-level defects in the critic

`controllers/rl/_diag_credit_reach.py` + `_d5_credit_reach_out.txt`. Reported FAIL. **Both verifiers
corrected to INCONCLUSIVE**, on the same fatal ground, and then both handed back something better than the
box asked for.

**Why the statistic cannot answer the question — it is an algebraic identity.** `:26-27` defines
`gap = Q(z,a,'avg') − G`. Since `G_t = r_t + γG_{t+1}` and `Q_t = r_t + γQ_{t+1} − td1_t`, we have
`gap_t ≡ −Σ_{k≥t} γ^(k−t)·td1_k` **exactly**. The curve can only report *where the one-step residuals
sit*; it carries no independent horizon information. And the residual mass is at the terminal, where
`terminated = 1` and `_td_target` returns `r_T` with **no bootstrap at all** (`agent.py:299-304`) — so
bootstrap length cannot be the cause of the thing the box was watching for.

**The pre-registered wrong-way signature DID fire, and it is still not evidence for C-6.** On c96's kill
cell the gap is **82% of |G| at 1-2 s and 76% at 2-4 s** with `P(Q<G) = 1.00` in every bucket, and
gap p50 = **−128.09** at 3-6 s on trajectories that DO end in a kill. Four things kill the horizon
reading:
- The gap is **maximal AT the terminal and shrinks outward** (−172.42 at 0-0.5 s, |gap|/|G| = 1.03, →
  −120.20 at 4-8 s). A credit-reach failure needs the opposite shape, and the harness pre-registers
  exactly that at `:118-122`.
- It **decays as γ^n**, at 0.99-1.18× of a pure γ^n decay of a single terminal mis-fit at k = 12/24/48/96.
- The one-step Bellman residual is **+0.002 (p50) on 739 of 745 non-terminal steps**.
- **Both D6 controls show the identical γ^n structure with the sign FLIPPED on a deck** (c95 asp55
  r_T −199.91, Q_T −65.51, td1_T −134.40; c87 asp55 the same, |gap|/|G| 0.75 → 0.20 monotone,
  `P(Q<G) = 0.00`). Every cell with **no** terminal spike **passes** the box's <30% clause (c96 asp55 0.10
  at 1-2 s; c95 asp30 0.24; c87 asp55 0.23); every cell **with** one fails. The failure tracks the
  presence of a large terminal reward, not the length of the bootstrap.

⇒ **C-6 (bootstrap length) stays closed, now on positive evidence.** What actually fails is
**terminal-reward magnitude representation**, and that is a different, live, unnamed defect. Two forms of
it, both measured rather than argued, and both new to this record:

**NEW MECHANISM 1 — decoder compression. The critic structurally cannot represent the kill reward.**
`functional.py:53-67` `two_hot_inv` returns `symexp(E_p[bin])` — an expectation taken in **symlog space**,
i.e. a log-mean, not the conditional expectation. Measured at the training latent on c96's kill cell: the
Q posterior puts **0.56-0.86 of its mass above +50** with modal bin **+147.41** (R_KILL = 150,
`constants.py:213`), and the shipped decode returns **+4.5 to +46.2, median +35.4**. The critic *knows*
about the kill and the decoder throws it away. This is the standard TD-MPC2/DreamerV3 two-hot symlog
decode, so it is not a typo — it is a design property whose bias is enormous **here specifically**,
because R_KILL = 150 sits far out on the symlog tail while every dense term lives near 0. **This is a
concrete, cheap, testable lever and it is the first one in this program that is not a reward constant or
an exploration knob.**

**NEW MECHANISM 2 — terminal-transition gradient starvation.** Measured by executing
`buffer._sample_starts` over 204,800 starts: with `slice_len = H+1 = 4` and the `_first` rejection
(`buffer.py:142-167`), the terminal stored index is reachable from **exactly one** slice start and always
lands at horizon position **t = 2**, drawn 2,107 times vs 6,124 for a mid-episode index — total ρ-weight
**527 vs 3,594 = 6.82× less gradient weight**. `_diag_credit_reach.py:383`'s "same ρ" assumption is wrong.
The one transition that carries the kill signal is the one the sampler under-weights ~7×.

**Three further defects on the box as run, so nobody re-derives its numbers:**
- **Evaluation-point artifact, and it is the headline number.** `:262-269` reads the terminal transition at
  `encode(o_{T−1})`. Training never touches it there — it always lands at t=2, i.e. a **2-step latent
  prediction**, not an encode.
- **The pre-registered 3-6 s window is mis-sized against its own cell.** c96's kill episodes are
  4.96-5.46 s, so "3-6 s before the kill" **is the episode opening** (n=319 of 745 pooled steps, 43%), the
  >8 s bucket is empty, and that is exactly where Q^μ-vs-greedy divergence is maximal.
- **The docstring's "no policy gap by construction" (`:73-79`) is false in mechanism.** `_td_target`
  bootstraps with `self.model.pi(next_z)[1]` — index [1] is the tanh-squashed **sample**
  (`world_model.py:96-98`), not [0] the mean. So Q's fixed point is V^{π_σ} while G is realised by π_mean.
  Separately, pessimism was reported **per backup** (`:475-478` prints Q(avg)−Q(min) p50 +0.094); its
  fixed-point deficit is that × Σγ^k = 92.6 over a 124-step kill episode, i.e. **−16.8 to −36.3**, which
  fully brackets the −17.7 / −27.5 / −20.8 residual left at k = 24/48/96.
- **`_estimate_value` cannot support "n-step has nothing to add":** it rolls `cfg.horizon = 3` latent steps
  = **125 ms**, two orders of magnitude short of the 1-8 s the box is about, and on the only kill cell its
  agreement with `Q(z, π(z))` is **level-only** — corr **+0.1778**, against +0.9925 to +0.9992 on all five
  spike-free cells.
- **The D6 controls are vacuous on the load-bearing cell:** neither produced a single kill in either cell,
  so the kill-cell curve rests on n = 6 episodes of one checkpoint on one cell.
- **Auditability:** D5 shipped no clean extract. Its three report blocks sit at lines 67,758 / 135,600 /
  203,455 of a 203,551-line, 6.3 MB file that is ~99% JSBSim banner spam. Extract before citing.

---

#### D6a — FAIL, and the leg is a DERIVATION, not the depth curve

`controllers/rl/_diag_latent_depth.py` + `_d6a_latent_depth_clean.txt`. Reported FAIL. **Both verifiers
corrected to FAIL and both replaced the reasoning.** The verdict is right; the depth curve it was resting
on cannot support it, and the box's own headline statistic is now banned.

**What actually kills D6b, and it is arithmetic that needs no measurement.** At `horizon = 3`,
`torch.pow(cfg.rho, torch.arange(H))` (`agent.py:317`) is evaluated at **k ∈ {0,1,2} only**, and the
consistency loop is `for t in range(H)` (`:325-327`). All three checkpoints print **H=3, ρ=0.5,
mpc=False**. So the weight vector is literally **[1.0, 0.5, 0.25]**, nothing in the stack ever rolls the
latent past depth 3 = **125 ms**, and depths 4-30 carry weight **exactly 0 at ρ=0.5 and at ρ=1.0 alike.**

⇒ **`rho: 0.5 → 1.0` at H=3 buys ZERO depth by construction.** What it actually does is a set of silent
global rescales — and this is the near-miss worth recording: it would have shipped as "one edit, single
variable" and been **four coupled coefficient changes**:
- ρ-weighted total **1.750 → 3.000 (×1.714)** ⇒ effective `consistency_coef` **20.0 → 34.29**, and
  `reward_coef` / `value_coef` scaled the same way;
- the depth weights **flatten** [1, 0.5, 0.25] → [1, 1, 1];
- the un-ρ-weighted **termination** share is **diluted ×1/1.714**;
- the actor loss is rescaled **×2.133**.

**The box's own statistic is banned from here on.** `ratio = predL2 / persL2` (`:32, :317`) is a
**state-velocity statistic, not a model-quality statistic** — it is driven by its denominator. On c96,
chase/asp30 vs chase/asp55: `predL2@d1` = 0.2053 vs 0.2121 (the model's error is 3.3% apart) while
`persL2@d1` = 0.5177 vs 0.0283 (**18.3× apart**). Worse, the denominator goes periodically near-zero: on
c95 chase/asp30 `persL2` alternates with the **parity of depth** (0.2080, 0.0608, 0.2081, 0.0587, …) — a
2-control-step (83 ms) latent limit cycle — while `predL2` grows smoothly, so the ratio alternates for
reasons that have nothing to do with the model. And the summary line "first depth where median ratio ≥
1.0 (stops beating persistence)" (`:337, :352`) assumes monotonicity the data **invert**: the ratio is
*decreasing* in depth on 6 of 9 cells (c96/asp55 7.450@d1 → 2.021@d30; c87/headon 16.050 → 5.158), so
nothing "stops" at the printed depth.

**Neither pre-registered branch is actually satisfied**, so "error at 10-30 is already low" is not a leg
the FAIL can stand on. Branch 1's own conjunction holds in **1 of 3** c96 cells. The three normalisations
give **three different orderings** for c96 vs c95 at chase/asp30, depth 30: raw `predL2` 1.2619 vs 0.8115
(c95 is 1.55× *more* accurate); `predL2/randpairL2` 0.331 vs 0.345 (a dead heat); `ratio` 0.426 vs 2.433
(c96 looks 5.7× better). **No cross-checkpoint model-quality claim is identified by this harness.** Also:
`predL2/randpairL2`'s normaliser is per-checkpoint and per-cell and swings 2.349 → 4.812 (**2.0×**), so it
is not comparable across checkpoints either — which was the only reason it was computed for the controls.

**Four measurement defects to inherit, not the numbers:** (i) episode-**phase** asymmetry, the banned
class — c96/chase-asp30 is 4 kill episodes of 5.3-6.3 s with `n_starts = 146` against controls flying
60.0 s timeouts with `n_starts = 1200`, an ~8× imbalance in the one comparison a reader will make;
(ii) the controls' reward-head errors are **scored against a reward function they never saw** — B1's
honest-WEZ edit added `and rg.rng <= C.LETHAL_RANGE_M` to `in_solution` (`reward.py:415-420`)
unconditionally, with no version flag, and c96 is the first run after it; (iii) precision exceeds the
sample — 4 episodes/cell, and re-flying c96/chase-asp55 greedy at a different seed moves `predL2@d1`
0.2121 → 0.2752 (+30%) and `@d30` 1.6867 → 2.2896 (+36%), though the sign and ordering are seed-robust;
(iv) a tail-dominated mean is printed unguarded (rMAE mean 3.3825 vs p50 0.5059 at c96/asp30 d30 — **do
not quote that mean**), and c96/headon carries 1 non-finite obs row in 13,678 whose filter covers only the
`randL2` pool. Also a declared but load-bearing deviation from the box's letter: the box says "on recorded
c96 transitions", but **no buffer exists** — all three `.pt` files are 77.6 MB, model-only — so the
harness rolls fresh greedy episodes. That is generalisation error on the deployed path, not the training
error the box asked about, and it is what creates the asp55 quasi-static confound.

**One thing genuinely worth keeping from the measured half**, on its own terms and not as a ρ argument:
c96's k-step latent error grows **sub-linearly** (`predL2 ∝ k^0.53-0.67` across all nine cells, 6.2-9.6×
over a 30× depth increase), and at depth 30 (1.25 s = 10× the trained depth) the prediction still retains
79-89% of the way from an unrelated-state pair to an exact match (cos 0.926-0.963 against an
unrelated-pair floor of 0.657-0.669). There is **no inflection at any depth**, and the model's advantage
over persistence strictly *improves* with depth in 8 of 9 cells. So the dynamics model is not the thing
that prevents a multi-second commitment from being represented — **the H=3 loop is**, and that is a
`horizon` question, which §3 row 2 closed on UTD cost, not a `rho` question.

⇒ **D6b is closed.** Not "weak" — closed, on arithmetic, for free, before any GPU time. If latent depth is
ever revisited it is `horizon`, with the measured 2.1-4.2× UTD cost paid explicitly, and never as "one
edit".

---

### ⚠ D3b HARNESS DEFECT — CMA-ES clips without penalty, so σ diverges (found mid-run, 2026-09-06)

Found by watching `sigma` climb monotonically 0.550 → 1.838 over 14 generations while `fit(mean)` sat at
−50k…−178k. **It is a defect in `_diag_true_mpc.py`, not a property of the merge.**

`CMAES.ask()` stores the **unclipped** samples in `self.pop`; `run_cell` then evaluates
`pop = np.clip(es.ask(), -1.0, 1.0)`; and `tell()` updates `xmean = w @ self.pop[idx[:mu]]`, `pc`, `ps` and
`Cov` from those same **unclipped** points. So the search fits its distribution model to points it never
evaluated. Beyond the box the objective is exactly flat (every out-of-box point shares its clipped image's
fitness), so σ meets no resistance in the normal direction and inflates; `xmean` can then walk outside the
box, at which point most of the 180 samples clip to nearly the same corner and the generation degenerates
to re-evaluating one point. That matches what the log shows: gen 6 found **+16,157.9** and gens 7–14 fell
back to +4…+1,204.

**The asymmetry that matters, and it is why this is cheap:** a broken *search* cannot invalidate an
*existence* result. Gen 6's genome was flown in the real sim, at real clipped values, across all 8 seeds,
and `hall` (initialised once per cell at :712, updated only on strict improvement at :728, re-flown at :744)
retains it. So the positive branch stands untouched. **Only the negative branch is weakened** — if a cell
returns no kill, the pre-registered wording must become *"no solution found at horizon H under objective J
**by a search with a known boundary-handling defect**"*, which is weaker still than the wording already
mandated. Do not read a D3b negative as evidence about the task.

**Fix before any re-run** (standard box-constraint handling, cheapest first): add a penalty
∝ ‖x − clip(x)‖² to the fitness that `tell()` receives, so the boundary is visible to the model; or store
the clipped points in `self.pop` so model and evaluation agree; or reparameterise the genome through
`tanh` so the map is smooth and unbounded → bounded. Not fixed mid-flight deliberately — the run already
holds the positive this box exists to find, and restarting would discard it.

---

### E2b GATE — the termination head is NOT starved, it is MIS-CALIBRATED ⇒ **E2b half (b) is dead** (2026-09-06)

`_diag_e2b_term.py`, committed and lint-clean. c96 as the run, **c95 and c87 as D6 controls in the same
invocation**; chase cell, sep 700 m, aspects 0/30/55, 20 episodes each, `--jitter 0.5`. 51.7 k–83.0 k ticks
per checkpoint. Pre-registered K1–K4 in the docstring before the run.

| ckpt | collected | n_pos | **AUC ± SE** | ticks p>0.5 | mean p ÷ sample rate | ÷ trained 3.03e-4 | peak **at** terminal |
|---|---|---|---|---|---|---|---|
| **c96 (run)** | 5,000,847 | 28 | **0.9818 ± 0.0253** | **0 / 51,718** | 0.0020× | 0.0036× | **0 / 28** |
| c95 (control) | 5,001,501 | 5 | 0.9904 ± 0.0436 | 0 / 81,329 | 0.0724× | 0.0147× | 0 / 5 |
| c87 (control) | 5,000,059 | 9 | 0.9436 ± 0.0769 | 0 / 82,953 | 0.0620× | 0.0222× | 2 / 9 |

**K2 (AUC ≤ 0.60) failed on all three, by 5–19 SE. K2 was the only outcome that justified half (b).** The
head is not signal-starved: it separates real terminals from fight states at AUC 0.94–0.99. Combined with
this box's own pre-check (`cfg.mpc = False` ⇒ nothing consumes the prediction), **E2b reduces to half (a)
alone — the terminal's 0.013% → 0.091% share of the reward/value supervision. That is not worth a 5M run,
and E2b should not be spent as one.**

**K4 failed on all three, robustly on BOTH denominators** (0.0020–0.0724× of the sample rate, 0.0036–0.0222×
of E2a's trained buffer rate) — the two-denominator construction was deliberate so the failure could not be
an artifact of which base rate I picked. The head is under-calibrated by **14×–500×**.

⚠ **CORRECTION to a claim this program has carried since 2026-09-04: "the termination head is dead" is a
MIS-DESCRIPTION, and it has been steering the diagnosis.** The recorded evidence was "max sigmoid 0.064 over
sampled fight states"; measured here the max non-terminal p is **1.06e-3 (c96) / 7.61e-3 (c95) / 1.81e-3
(c87)** — lower still, yet the head is simultaneously **near-perfectly ordered**. The accurate description is
**correctly ranked, catastrophically under-calibrated, and never crossing the `logit > 0` latch** — which is
what an un-ρ-weighted BCE on a 3.03e-4 positive rate produces. "Dead" implies uninformative; AUC 0.98 says
otherwise, and the two diagnoses point at completely different fixes.

**M5, which the pilot run forced into the harness and which is the entry's real find:** the head's peak
belief is **not on the terminal tick** — 0/28 for c96, 0/5 for c95, 2/9 for c87, with the median argmax at
**0.993–0.996 of the episode**, i.e. ~5–6 control steps (~0.2 s) *before* the end, after which p *falls*.
So the head has learned "a terminal is imminent" and then assigns the actual terminal state a LOWER
probability than its own immediate predecessors. **AUC alone cannot see this** — a high AUC only says
terminals outrank the median tick. Do not read K3 without M5.

**The D6 control did exactly what the rule exists for.** c87 predates every value-path change in this
program and shows the same signature (AUC 0.94, zero latch, 45× under-calibrated, peak off the terminal).
⇒ this is a property of **the architecture and the base rate**, not of anything the program changed, and
any future "we improved the termination head" claim must clear c87 in the same invocation.

### M7/M8 RESOLVED — **the whole table above was measured at a query point the head has never been trained on** (2026-09-06)

**Every number in the gate table queries `encode(obs)`. The termination head has literally never seen an
`encode()` output during training.** `agent.py:339` is `term_logits = self.model.termination(zs[1:])` — and
`zs[0]` is the *only* encoded latent in the batch; `zs[1:]` are all `next()` rollouts. Contrast
`agent.py:328`, `_zs = zs[:-1]`, which the reward and Q heads use and which **includes** `zs[0]`. So
termination is the one head in the model whose training slice excludes depth 0. This is **categorical, not
distributional** — not "few samples near the encoded manifold" but *zero*, by one slice index.

M7 re-asks p at the **same physical state**, varying only how its latent was produced: `encode(obs_j)` vs
`encode(obs_{j-d})` rolled forward d steps with the **actions actually taken** (so policy drift is excluded
too). M8 is the mandatory control — the identical ramp anchored mid-episode, where no terminal is near.

| depth | how the latent was made | M7 p50 **at the terminal** | paired ratio | ticks p>0.5 | M8 p50 **mid-episode** | paired ratio | ticks p>0.5 |
|---|---|---|---|---|---|---|---|
| 0 | `encode(obs_j)` — **never trained** | 7.894e-06 | — | 0/16 | 2.396e-07 | — | 0/16 |
| 1 | `next¹(encode(obs_{j-1}))` | 8.997e-05 | 12.1× | 0/16 | 2.479e-07 | **1.0×** | 0/16 |
| **2** | `next²(…)` — **E2a's trained depth** | **4.900e-04** | **62.7×** | 0/16 | 2.489e-07 | **1.0×** | 0/16 |
| 3 | `next³(…)` | **9.260e-01** | 101,217.6× | **13/16** | 2.475e-07 | **1.0×** | 0/16 |

**The control is what makes this a finding.** M8 is flat to three digits at every depth — ratio **exactly
1.0×**, 0/16 latching — so the ramp is not `next()` drift reading as "terminal" regardless of state.
Ratio-of-ratios at the trained depth 2 = **59.94×**. The head fires on *imminent death*, and only where it
was trained to. ⚠ This is also a measured counterexample to over-reading D6a's cos 0.926–0.963 at depth 30:
the encode/rollout latents are 0.93-cosine-similar and the termination head still reads them 62.7×
differently. **Cosine similarity is not the direction a given head reads.**

**What this overturns, in order of consequence:**

1. **"The termination head is dead" is REFUTED, not just mis-described.** At its trained query point p50 is
   4.900e-04 ≈ **1.6× E2a's trained base rate 3.03e-4** — roughly calibrated. K1's and K4's failures above
   are **query-point artifacts**. ⚠ Strictly, K4 tested the *mean* and M7 reports *medians*; the clean form
   is to re-run K1/K4 with the depth-2 query point, which is ~3 min. The direction is not in doubt (14–500×
   under-calibrated at depth 0 vs 1.6× at depth 2), the exact factor is.
2. **"The planner never latches a terminal" is FALSE.** At depth 3, **13 of 16 real terminals cross p > 0.5**,
   so `dead = (termination(z) > 0.0)` (`agent.py:290`) *would* fire. Its status changes from "true but inert"
   to **"false, and separately inert because `cfg.mpc = False`"**.
3. **The "dying is worth +100..170" planner defect stops being inert — but on inspection it is CONDITIONAL,
   and the condition is unmeasured.** ⚠ **Correction to what I first wrote in this entry:** I recorded it as
   a confirmed consequence. Re-reading `_estimate_value` (`agent.py:274-296`), the loop is **correct for an
   episodic MDP**: `G += discount·(1-term)·reward` adds the reward on the transition *into* the terminal
   while `term` is still 0 (`term` only latches on the *next* line, from the post-`next()` latent), and
   value-after-terminal genuinely *is* 0. So zeroing the tail bootstrap is not a bug by itself. The two
   branches:
   - if the **reward head predicts ≈ `R_DECK = −200`** on real death transitions, a terminating plan scores
     ≈ `−200·γ^t` against a non-terminating `≈ G + γ^H·Q`, Q ≈ −66.5 ⇒ **death is correctly avoided, no
     defect, and MPC-mode is back to "expect a null"**;
   - if the reward head predicts **≈ 0** there, the terminating plan drops the −200 *and* the negative
     bootstrap ⇒ death is worth **≈ +|γ^H·Q| ≈ +66.5** (not the +100..170 I have been quoting, which came
     from reading |Q| alone and double-counted nothing but my own arithmetic) ⇒ **elite set captured by
     death-seeking plans.**
   **E2a makes the second branch plausible but does not establish it** — the terminal transition is 0.013% of
   the reward/value loss, one per 4.3 updates, so the reward head has had little chance to learn −200. That is
   a *reason to measure*, not a measurement. **THE DECIDING TEST IS CHEAP AND MUST PRECEDE ANY MPC RUN:**
   query `two_hot_inv(reward(z_{T-1}, a_{T-1}))` on real death transitions at the **trained depth** (this
   entry's own lesson) and compare to −200. Same harness, ~10 min. See the new §6 box **E3**.
   ⇒ Standing status of the MPC cell: **"expect a null" is SUSPENDED, not retired** — it is now
   branch-dependent, and E3 picks the branch.
4. **K2 is untouched and half (b) stays dead.** AUC 0.94–0.99 at depth 0 already proved the head is not
   signal-starved; M7 only strengthens that (it discriminates *better* where it was trained). Nothing here
   revives E2b half (b), and the `cfg.mpc = False` pre-check still caps half (a) at the 0.013% → 0.091%
   share. **E2b remains closed by measurement, not by a run.**
5. **M5 inherits the caveat.** Its "peak belief is 5–6 steps *before* the terminal" was also measured at
   depth 0. Whether the peak sits *on* the terminal at depth 2 is **not yet measured** — fold it into the
   K1/K4 re-run above rather than citing M5 as a standalone fact.

⚠ **METHOD RULE, and it generalises past this box:** before diagnosing any auxiliary head, read the slice
index it is trained on and query it *there*. `zs[1:]` vs `zs[:-1]` is a one-character difference that moved
a measured statistic by **62.7×** and inverted the conclusion. This sits with the range-gating trap
(§9, c96) and the noise-floor trap (E2a K3) as the third instance of the same failure: **the harness
measured something real, just not the quantity the claim was about.**

---

### E3 RESULT — **the reward head cannot tell a kill from a death**, and my own pre-registration was wrong (2026-09-06)

`_diag_e3_reward.py`, committed and lint-clean. c96 as the run, **c95 and c87 as D6 controls in the same
invocation**; chase, sep 700 m, aspects 0/30/55, 20 episodes each, `--jitter 0.5`. Every residual is against
the reward the **env actually returned** (`ActionRepeat` sums, dense terms ride along), not against the
constants. All numbers below are at **depth 2 — the only depth the terminal is ever supervised at** (E2a).

| ckpt | collected | kills | kill pred@d2 | **% of +152** | deaths | death pred@d2 | **% of −200** | dense \|res\| |
|---|---|---|---|---|---|---|---|---|
| **c96 (run)** | 5,000,847 | 27 | **+111.77** | **73%** | 1 | −5.33 | 2.7% | 0.046 |
| c95 (control) | 5,001,501 | 4 | +16.78 | 11% | 1 | +0.18 | −0.1% | 0.046 |
| c87 (control) | 5,000,059 | 2 | +7.90 | 5% | 7 | **−0.19** | **0.1%** | 0.087 |

**Gate: K1 FAIL ×3, K2 FAIL ×3, K3 PASS ×3, K4 FAIL ×3** — which fires the pre-registered reading verbatim:
*K1 and K2 fail while K3 passes ⇒ the learned reward function is outcome-blind.* The head is accurate to
**0.046–0.087 reward units** on the dense terms it sees every tick and wrong by **~195–200** on the outcome.
⚠ c96's kill is a **near** miss, not a clean failure (|rel| 0.272 against a 0.25 gate, 11/27 within 25%) —
"73% priced" is the honest description, not "blind".

**⛔ MY PRE-REGISTERED PREDICTION IS REFUTED, and the truth is more specific.** I predicted that if data
density drove it, c87 — which decks 7 times here — would show the **mirror image**: death priced well, kill
badly. It does not. **c87 has 7× more deaths than c96 and prices death at 0.1% versus c96's 2.7%.** Death
pricing is **frequency-independent**. But kills *are* frequency-dependent, cleanly and monotonically across
three independent checkpoints: **2 kills → 5%, 4 kills → 11%, 27 kills → 73%.** So the effect is
**sign-specific, not density-specific**: the head learns the *positive* terminal in proportion to how often
the policy achieves it, and never learns the *negative* one at any frequency. That is a sharper claim than the
one I set out to test, and I only get to make it because the prediction was written down first.

**The headline, stated as the thing that is actually broken.** The predicted gap between killing and dying is
**117.1 (c96) / 16.6 (c95) / 8.1 (c87)** against a true gap of **352.9**. On c87 the model compresses the
entire outcome axis by **43×**. To this reward head a kill and a death are very nearly the same event.

**Depth-2 specificity — this is now measured three times and it is the strongest evidence for the M7/M8
mechanism.** The kill ramp *peaks at the trained depth and is worse on BOTH sides*:
c96 +1.13 / +2.07 / **+111.77** / +35.95, c95 +1.17 / +1.74 / **+16.78** / +4.21,
c87 +2.21 / +2.94 / **+7.90** / +4.23 (depths 0/1/2/3). A non-monotone peak at exactly the supervised depth
cannot be `next()` drift. Meanwhile **death is FLAT across depth** (c87 −0.18/−0.18/−0.19/−0.19; c95
+0.16/+0.19/+0.18/+0.24) ⇒ it has not been learned in *any* form, at any query point.

**K4 / the MPC branch — the death-seeking defect is CONFIRMED, and the magnitude I first recorded was wrong.**
Advantage-of-terminating on real deaths: **+83.07 (c96) / +66.22 (c95) / +56.48 (c87), attractive in 9 of 9
real deaths across three checkpoints.** So the §9 M7/M8 branch resolves to the bad half: a predicted terminal
drops both the −200 and the negative bootstrap. ⚠ The magnitude is **+56 to +83**, not the **+100..170** this
program has carried since 2026-09-04 — my corrected estimate of ≈+66.5 was right and the original was not.
⇒ **MPC must not be run until the terminal is priced. "Expect a null" is now formally RETIRED, with a
measurement behind it rather than an inference.** Note also that MPC could not even prefer a kill *as a kill*:
c87's predicted kill (+7.90) beats continuing only because continuing is worth −46.36.

⚠ **SCOPE — the sentence this entry must not be allowed to become.** This is **not** "everything optimises a
blind reward". `_td_target` is called on the tuple straight out of `buffer.sample()` (`agent.py:308,:313`), so
**the critic's TD target uses the ground-truth reward and is NOT blind.** An outcome-blind reward head costs
exactly two things: (1) MPPI is unusable — moot at `cfg.mpc = False`, decisive if it is ever switched on;
(2) of the three gradients shaping the shared latent (reward-prediction, Q, consistency), the
reward-prediction one carries **no outcome information**, so the encoder is organised around the outcome only
via Q and consistency. Since `act()` is `pi(encode(obs))` the policy reads that encoder, which makes this a
**representation** claim — the one family that has ever moved this program. ⚠ It does **not** license "the
policy cannot see death risk": Q sees −200 through the buffer. Do not upgrade this to a behavioural claim
without a behavioural measurement.

**D6 did its job again.** c87 predates every value-path change in this program and shows the same signature
⇒ **architectural and base-rate-driven, not something this program caused.** Any future "we fixed the reward
head" claim must clear c87 in the same invocation.

**What it authorises, and what it does not.** It authorises fixing the **supervision** — E2b form (i),
oversample terminal-hosting starts, sampler-only, already costed, under the c88 tripwire. It does **not**
authorise touching a reward constant (γ and the reward constants are a coupled pair — c88/c89). ⚠ And note
E3 partially **revives E2b's half (a)**, which the gate had reduced to a 0.013% → 0.091% curiosity: the
terminal's share of the *reward-head* loss now has a measured consequence (the 43× outcome compression) rather
than only a share-of-loss argument. It is still not worth a 5M run on its own; it is worth landing the moment
anything else touches `_sample_starts`.

⚠ **HARNESS DEFECT found and fixed mid-box, recorded because it nearly cost the result:** the pilot ran with
`KILL_EVENTS = ("killed",)` while the sim emits `"kill"` (`reward.py:503`), so the aggregate printed
`kills 0` while the per-episode table was full of them. The fix reads the producer instead of guessing the
string, adds `left_bubble` as a third outcome class, and **asserts that no episode falls outside a reported
class**. Also fixed in the same pass: K4 originally required `adv <= 0` for *every* class, which would have
failed a healthy head for correctly pricing kills — it now tests DEATH/EGRESS only. Same family as the
`zs[1:]` slip: **the harness measured something real, just not the quantity the claim was about.**

---

### D2d RESULT — **coloured exploration is LANDED and does NOT CONVERT** (2026-09-06) — NEGATIVE

Two matched 5M arms, sequential, single delta, warm from c96 on `chase-t3-killgate`, judged out-of-band:
`bfm_detect`, cell **chase**, sep 700 m, `--max-time-s 60`, `--jitter 0.5`, aspects **0 / 30 / 55**,
8 eps/aspect = 24 eps/arm, `LETHAL_RANGE 600 m` (D5 on `tLeth`, not the banned `tHit`), **all three
checkpoints in one invocation.**

| arm | ckpt | collected | stage_idx | kills/24 | asp 0 / 30 / 55 | decks | t/out | dealt |
|---|---|---|---|---|---|---|---|---|
| **B (treatment)** | `c97-ou` τ=36 | **5,000,198** | 3 | **11** | 4 / 7 / 0 | 0 | 13 | 70.8 |
| A (control) | `c97-white` | **4,814,165** ⚠ | 0 | 5 | 3 / 2 / 0 | **6** | 13 | 36.1 |
| incumbent | `c96` | 5,000,847 | 1 | 10 | 2 / 8 / 0 | 0 | 14 | 47.2 |

**The verdict is a NULL against the incumbent, and that is the comparison that counts.** OU 11 vs c96 10,
Fisher two-sided **p = 1.0000**; at aspect 30 (the standing mandatory cell) OU 7 vs c96 8, **p = 1.0000**.
Five million steps of the one exploration change that D2b/D2c proved actually reaches the actuator bought a
kill count indistinguishable from its own starting checkpoint.

**And the mechanism claim fails harder than the score claim.** `D1 displacement_roll/ep = 0.00` and
`D2 yoyo/ep = 0.00` **on all three arms, at every aspect, without exception** — and `D3 turning_room/ep` is
*highest on the incumbent* (c96 0.50 vs OU 0.25 vs white 0.21). So the seam is real (P(1.5 s coherent roll
deviation) 0.000 → 0.256, N_eff 8.2 → 1.2, measured closed-loop in D2c) and the greedy policy it trains does
**not one** displacement roll or yo-yo. ⇒ **"conversion is still unproven — that's a run"** is now answered:
**it does not convert.** Getting coherent out-of-plane *exploration* into the buffer is not sufficient to get
out-of-plane *behaviour* out of the policy.

⚠ **DO NOT READ THE 11-vs-5 AS THE RESULT.** It is the comparison the box was designed around and it does not
survive contact: **Fisher p = 0.1246**, i.e. not significant at n = 24, despite looking like a 2.2× win.
Worse, **Arm A is not a usable control** — it differs from Arm B in *three* ways at once: (1) the intended
lever, (2) `collected 4,814,165 / 5,000,000 = 96.3%`, a **3.7% shortfall** (⚠ this is exactly the
`verify collected ≈ 5M` check earning its keep — I nearly wrote the arm up as matched), and (3) it **never
left curriculum stage 0** while Arm B reached stage 3. Any of the three could produce the gap. The only two
significant cells in the whole table involve Arm A — decks 6 vs 0 (**p = 0.0219**) and asp30 2 vs 7
(**p = 0.0406**) — and both say only *"the white arm trained badly"*, which is a statement about Arm A, not
about colour.

⚠ **`stage_idx 3 vs 0` is the most tempting number here and it is INADMISSIBLE as performance.** A 3.7% step
shortfall cannot explain zero stage advances in 4.81M steps, so something real differs — but the advancement
gate is the anchored probe, and [[selfplay-gate-lies]] is the standing finding that this gate scores against
weak copies and lies. The program's rule is **judge only the out-of-band greedy verdict**. Record stage_idx as
a training-dynamics observation and nothing more; the greedy verdict on the same checkpoint is p = 1.0000.

⚠ **D5 is a textbook base-rate inversion — quote the base rate or do not quote D5.** `P(lethal|event)` orders
**OU 0.667 > white 0.600 > c96 0.417** while the *ratio* orders **c96 85.00 > white 19.38 > OU 11.83** —
exactly backwards — because the base rates differ 11× (0.056 / 0.031 / 0.005). Both orderings are also built
on 5–12 lethal windows, far under any usable n. This is the third time the D5 ratio has had to be discarded;
it stays on the banned-statistics list.

**Scope limits to state plainly.** (a) **Chase only** — no merge/head-on cell was run, so this says nothing
about the cell the program actually targets; the exemption that licensed a chase-only D2d was about the
*training signal*, and it applies, but the verdict inherits the narrowness. (b) There is **no perfectly
matched control**: Arm A is degraded and c96 differs in stage_idx, so "null vs incumbent" is the strongest
sound reading, not "null vs a twin". (c) 24 episodes/arm resolves nothing smaller than a ~2× effect.

**What this closes.** The exploration family is closed **as a scoring lever**: amplitude was closed by D2a
(σ_ail 0.67–0.90 on the executed path), correlation was landed by D2b/D2c, and D2d shows the landed version
converts to neither kills nor BFM events. ⇒ **the limiter is not exploration.** Combined with D6a (the
dynamics model is not the limiter) and E3 (the reward head cannot tell a kill from a death), the surviving
candidate is where this program's only two wins have ever come from — **representation**.

---

### D3b RESULT — **the merge kill is REACHABLE; the out-of-plane kill is not** (2026-09-06)

`_diag_true_mpc.py`, CMA-ES over an 18-dim open-loop bank schedule (6 segments × 3 channels, `roll_gate=True`
so out-of-plane content is directly representable), popsize 180 = 10·dim, 40 generations, fitness = **mean over
8 `sim.seed` values** (never a max over the stochastic objective), 24.0 s / 576-step horizon. Two cells,
~15,187 rollouts and 6,080 s of wall per cell. **Prechecks as mandated: replay determinism
`max|div| = 0.000e+00` over 576 control steps, cross-seed `max|div| = 0.000e+00` ⇒ seed-skip VALID,
throughput 1,352 / 1,412 sim ticks/s (3.70 / 3.54 ms per control step)** — which confirms the *corrected*
11.1–11.3× realtime figure and not the retracted 340×.

**merge1500** (lateral 1500 m, sep 3000 m, aspect 30°, perch 2000 ft) — best genome re-flown over every seed:

| arm | kills/8 | hp dealt | `tLeth` | `rngMn` | `dalt` | rounds |
|---|---|---|---|---|---|---|
| **SEARCH** | **5** | **99.0** | **0.521 s** | **8 m** | −1293 m | 79 |
| `recorded` (hand-coded one-circle lead turn) | 0 | **0.0** | 0.000 | 906 m | +38 m | 0 |
| `recorded_two` (same arm, two-circle) | 0 | **0.0** | 0.000 | 642 m | +74 m | 0 |
| `inplane` (pursuit) | 0 | **0.0** | 0.000 | 267 m | −1932 m | 0 |

The 5 kills finish in **172–173 steps ≈ 7.2 s** of the 24 s horizon at 100.0 HP with 64–68 rounds; the 3
non-kills fly the full 576 steps, still deal **96–98 HP**, and dive **−2602 m**. **Every control deals exactly
zero.** So fix (4) is satisfied with power in both directions — the search beats the recorded lead turn *and*
beats in-plane pursuit from a start where in-plane provably fails. ⇒ **A physics-respecting, gun-lethal
solution to the neutral merge EXISTS and is findable.** ⚠ It is **in-sample by construction** (CMA-ES optimised
the mean over exactly these 8 seeds and the verdict re-flies on the same 8), which is the correct frame for an
*existence* claim and is **not** a generalisation claim. Do not quote 5/8 as a hit rate.

**merge800** (lateral 800 m, aspect 15.5°): **nothing.** 0 kills, `tLeth` 0.083 s, 16 lethal ticks, best
fitness never exceeded +3.8 in 40 generations, and the best genome is byte-identical across all 8 seeds
(`rngMn` 288 m, `dalt` −535 m, 16 rounds) — it never gets close enough for gun dispersion to matter. ⇒
convertibility is **geometry-dependent**: 1500 m of lateral offset is convertible at this horizon, 800 m is
not, which is the direction [[curriculum-94-headon-lateral]] predicts (lateral offset buys turning room).

**⛔ THE PRE-REGISTERED VERDICT IS THE NEGATIVE, AND IT STANDS ON BOTH CELLS.** The box required *"≥1 kill
carrying a loaded D1 or D2 event and nonzero `tLeth`"*. Detector on the SEARCH arms: **D1 = 0, D2 = 0,
D2c = 0** on both cells (merge1500 D3 = 3, `chiMx` 178.7°, `nzMx` 7.4; merge800 D1 = D2 = D3 = 0). So the
positive branch did **not** fire and the only admissible statement is the one the harness printed:
**"no solution found at horizon H under objective J"** — it bounds achievability from below and **motivates no
task change, no reward change, and no benchmark change.** The conclusion the box reserved for a positive —
*"a feasible out-of-plane demonstration exists and the search space has been learner-side all along"* — is
**NOT licensed.**

⚠ **But the box conflated two questions and I must not let the bundled negative hide the split.** "Is the merge
kill reachable?" → **yes**. "Is an *out-of-plane* merge kill reachable?" → **not found**. Both halves are
informative and they point opposite ways: the first says the task is winnable and the ceiling is learner-side
*for scoring*; the second says nothing in this search space rewards BFM enough to find it.

**The search STALLED at the kill, which is the sharpest part of the result.** merge1500 hit its hall-of-fame
best at **generation 6** (`fit +16,157.899`, the 5-kill genome) and **never beat it in the remaining 33
generations** — per-generation best decayed to +1,879 by gen 39 while σ drifted **1.58 → 1.84**. ⇒ the kill was
findable in **6 generations of an 18-dim search**, and 33 further generations with out-of-plane content fully
representable produced **zero** loaded displacement rolls and **zero** clean yo-yos. ⚠ Read with the honest
qualifier: objective J pays for kill / `tLeth` / altitude and **not** for BFM, so this shows BFM is **not
necessary** to convert this merge — it does **not** show BFM is useless. ⚠ And the CMA-ES
**clip-without-penalty defect** (σ inflating because clipped genomes are free) means the search is *weaker*
than nominal, which strengthens the negative's caveat — a better-conditioned search might find out-of-plane
solutions — while leaving the positive intact, since a found kill is a found kill.

**⇒ THIS IS THE THIRD INDEPENDENT MEASUREMENT THAT KILLING HERE DOES NOT REQUIRE BFM,** and the strongest of
the three because it is not a learner that failed to acquire BFM: (1) c95 chase reaches the gun envelope at
D1 = D2 = D3 = 0.00 (§9, policy-side); (2) c96's honest-WEZ 45 kills carry zero BFM (B1); (3) an unconstrained
trajectory optimiser handed 15,187 true-sim rollouts converts the neutral merge with zero BFM and never needs
any. **The program's stated goal (learn displacement rolls / out-of-plane / turn-circle geometry) and the
program's scoring metric (kills) are now measured to be close to independent in this environment.** That is a
statement about the environment, not about the learner, and it is the single most important thing on this
board for deciding what to do next — see §10.

⚠ **`runs/d3b/d3b_best.npz` (173 control steps) is written and NOTHING CONSUMES IT.** `update_pi`
(`agent.py:241-259`) has no BC/AWR/demo term and `buffer.py:42-47` has no demo label, so splicing it would
train only encoder/dynamics/reward/Q and the maneuver shape would never reach the deployed controller
([[bfm-learnability-six-disproofs]]). **The demo is necessary, not sufficient; the BC channel is a separate
unbuilt change.** Do not claim D3b unblocks D-B.

---

### F2 RESULT — **the positive control fails, and D1 on the merge is measuring the PASS** (2026-09-06)

Two boxes in one invocation: **F2a** the detector positive control (never run before), **F2b** the
`me_dalt_ft = 0` re-fly that D3b left as a written falsifiable prediction. Harness
`controllers/rl/_diag_f2_poscontrol.py` + the addendum `controllers/rl/_diag_f2_verify.py`; reports
`runs/f2/f2_report.txt` and `runs/f2/f2_verify.txt`. 15 arms × the 8 D3b seeds (11/28/45/62/79/96/113/130),
`jitter = 0`, `a[4]` overwritten with the same `_trigger(rel)` oracle on **every** arm so no arm can win on a
different fire rule. Chase 60 s / 1440 steps, merge1500 24 s / 576 steps.

**F2a pre-registered three exclusive outcomes: (A) flew + seen ⇒ validated; (B) flew + unseen ⇒ too strict,
every zero-BFM reading suspect; (C) did not fly ⇒ library cannot generate it, detector still unvalidated.
The measurement returned a FOURTH thing the pre-registration did not enumerate, and that is the result.**

#### Q1 — the library DOES fly a real maneuver on the merge (this half is clean)

| arm | `nzP50` | `nzMax` | `frLoad` | `chiMaxL` | **`chiHold`** | `gamSwg` |
|---|---|---|---|---|---|---|
| `merge:lead_turn` | **6.11 g** | 7.26 | **0.978** | 178.3° | **1.958 s** (3.9× the 0.5 s bar) | 45.6° |
| `merge:coach` | **7.32 g** | 6.99 | 0.927 | — | **2.458 s** | 128.4° |

`chiHold` is the raw D1 ingredient measured independently of every clause: longest contiguous
`|chi| > CHI_OUT_DEG ∧ nz > NZ_LOADED`. So **Q1 = YES**: `bfm.LeadTurn` and `BFMCoach` fly a sustained,
heavily-loaded, out-of-plane pull. ⇒ This **narrows** A2's "the stick library cannot fly a yo-yo at any
threshold" — it cannot fly a *yo-yo* (**D2 = D2c = 0 on all 15 arms × 8 seeds = 120 episodes; nothing in this
entire run demonstrates that D2 can fire at all, so D2 remains unvalidated in the positive direction**), but
it does fly a loaded out-of-plane lead turn. This also corrects one A2 sub-claim — A2 recorded the lead turn's
out-of-plane content as *"shallow (30–45°), not the 60° a displacement roll needs"*; on **merge1500** it
reaches `chiMaxL` **178.3°** and holds past the shipped 60° bar for 1.958 s. **The difference is the cell, not
the stick**, which is the same lesson as the entry finding at the bottom of this section: a primitive's
measured content is a property of the geometry it is flown at.

#### Q2 — the detector fires, and the "yes" carries NO information

D1 = 8/8 on four merge arms. The addendum classified every passing event by how clause (d) was satisfied:

| arm | HP dealt | rounds | aspect | **ata** | rng (min) | closure | clause (d) exit | D1 |
|---|---|---|---|---|---|---|---|---|
| `merge:lead_turn` | **0.0** | — | 135.7 → 71.7 | 75.7 → **154.4** | 400 → 860 (**339**) | +253 | valid, 29.9° | **8** |
| `merge:coach` | **0.0** | — | 159.0 → 69.9 | 22.1 → **138.9** | 626 → 989 (**299**) | +416 | valid, 29.1° | **8** |
| `merge:FLYBY(d1exists)` | **0.0** | — | 175.8 → 25.3 | 12.5 → **171.3** | 890 → 349 (**97**) | +453 | **SENTINEL, χ invalid** | **8** |
| `merge:d3b_perch0` | **0.0** | **0** | 170.1 → 62.1 | 22.9 → **156.0** | 1881 → 816 (**407**) | +411 | valid, 25.8° | **8** |
| `merge:d3b_perch2000` | **99.0** | 64–68 | — | — | — | — | — | **0** |

**All four D1-positive arms are the same event: a merge pass.** Every one ends with the bandit *behind* the
jet (**ata 139–171°**) and range **rising off a 97–407 m minimum** at +253..+453 m/s closure. Two independent
routes to the same vacuous conclusion — the fly-by satisfies (d) via the **χ-validity sentinel** (`chi = 0.0`
returned when ata < `CHI_VALID_ATA_DEG` = 5°, i.e. bandit dead ahead), while the other three satisfy it with a
**valid** χ measured against a LOS pointing *over the shoulder*.

**Both load-bearing clauses are supplied by the geometry of the pass, for free:**
- **(d) "rolled back in" has no requirement that the bandit still be in front.** A small valid `|chi|` at
  ata 156° is not re-attacking him; it means the lift vector happens to lie in the plane containing a
  *rearward* LOS.
- **(f) "aspect improved by ≥15°" is automatic on any head-on pass**, because aspect is measured off the
  *target's tail*: fly past a jet you approached head-on and aspect necessarily collapses. `d_aspect` −64° to
  −150° here is what the pass hands you, not what out-turning bought.
- **(e) "same side" does not catch it either** — a clean fly-by down one side never flips `ata_az`, so
  `crossed_nose = False` on all four.

⇒ **On the merge cell D1 is anti-correlated with lethality: it fires 8/8 on four arms that each deal exactly
0.0 HP (one of which fires 0 rounds and dives 2.4 km) and 0/8 on the only arm that kills.** The killer never
produces a ≥0.5 s loaded out-of-plane hold: 11 of its 14 candidates die on `hold_s`, 3 on (a) lookback.
**Every "D1 on the merge/head-on" number in this document is therefore uninterpretable in both directions and
must not be cited as BFM evidence.**

#### The chase cell gives the OPPOSITE instrument verdict — and there "zero BFM" survives

D1 = 0 on all ten chase arms, with a single sole rejection reason. chase30's `high_yoyo`/`barrel`/`low_yoyo`/
`coach` each produce 8 holds of **0.542 s at `chi_peak` 73.5°, `nz_peak` 3.55 g** — clearing the raw
ingredient — and **all 8 are rejected by (f) at `d_aspect` −6.5° against the −15° bar**, zero rejected by
anything else. chase0's `lag_roll` produces a **1.333 s hold at `chi_peak` 144.6°, 5.25 g** rejected by
**(c) closing at −7 m/s** — a knife-edge sign test on a noisy quantity killing a real maneuver by 7 m/s.
On the chase a pass *cannot* hand you free aspect (co-heading, already low aspect), so (f) is doing honest
work there and the primitives genuinely do not buy angles. **The learner-side "zero BFM on the chase"
readings (c95 D1 = D2 = D3 = 0.00, D2d's three arms) stand as written** — and are if anything reinforced: even
the permissive merge-side D1 that a *pass* can satisfy never fired for any policy, meaning the policies never
flew a loaded pass at all, consistent with the unloaded −14,807 m spirals.

#### The minimum instrument repair (~~not yet made~~ **LANDED AND AUDITED — see F7 RESULT, §9**)

> ✅ **All four steps are in `bfm_detect.py` behind switches as of 2026-09-07, and F7 measured them
> arm-by-arm.** Result: the four 0.0-HP merge arms go **8/8 → 0/8**; steps **1–3 alone are sufficient**;
> **step 2 alone and step 3 alone are each sufficient**, so the repair is over-determined. ⚠ Two of my
> claims below are wrong as written: **step 1 alone is NOT sufficient** (it is a filter inside the exit
> search, so on `lead_turn`/`coach` it *relocates* the exit to just inside the 120° bar instead of
> rejecting), and **step 4 is not a loosening** (it rejects `lead_turn`'s pass outright). Read F7 RESULT
> before touching any threshold here.

1. Gate clause (d) on `chi_valid` **and** a forward-hemisphere bound on ata (≤ ~90–120°) — kills both the
   sentinel route and the over-the-shoulder route.
2. Require range to be **non-increasing** at the (d) exit — a pass always exits on the opening side of its
   own minimum.
3. Replace or augment (f) with a quantity a pass cannot supply: **ata** gain, or turn-rate advantage, rather
   than aspect off the target's tail. Keep (f)'s magnitude idea, drop aspect as its carrier.
4. Give (c) a small negative tolerance, or evaluate closure over the hold rather than at the single entry
   tick, so a −7 m/s reading does not discard a 1.333 s / 144.6° / 5.25 g displacement.

⚠ ~~Until 1–3 land, **do not run any box whose success criterion is a D1 count on a merge or head-on
cell.**~~ **REPLACED 2026-09-07 by F7 (§9): 1–3 have landed and are measured. A merge/head-on D1 count is
now admissible as evidence of ABSENCE and not of presence** — the repaired D1 has no positive control on
any arm, so a nonzero merge D1 requires a per-event autopsy (exit ata, exit closure, both currencies)
before it is believed. Never cite a *pre-repair* merge/head-on D1 count at all.

#### F2b — CONFIRMED. The D3b merge kill is an artifact of the +2000 ft perch

Same invocation, same seeds, same env build. The only change is `me_dalt_ft`:

| arm | kills/8 | hp | `tLeth` | `rngMn` | rounds | `dalt` |
|---|---|---|---|---|---|---|
| `d3b_perch2000` (baseline) | **5** | **99.0** | **0.521 s** | **13 m** | 64–68 | −1293 m |
| `d3b_perch0` | **0** | **0.0** | **0.000** | 407 m | **0** | **−2422 m** |

The baseline **reproduces the recorded D3b row exactly** (5/8, 99.0 HP, `tLeth` 0.521 s) ⇒ the harness is
sound, and this is a clean self-check rather than a re-derivation. At perch 0 the result is **identical on all
8 seeds** — correct, and itself a check: with **0 rounds fired**, gun dispersion (the only thing `sim.seed`
drives) has nothing to differentiate, so byte-identical rows are the expected output.

**The genome does not merely fail to kill at co-altitude — it never achieves a firing solution at all.** The
honest trigger never opened once in 24 s across 8 seeds. ⇒ D3b's written prediction is confirmed and its
mechanism is now supported by a controlled swap: `BREAK_BANK_RAD = 85°` (`constants.py:123`,
`opponent.py:425-461`) is a hard constant with **no LOS-plane term**, and it is the **+2000 ft perch** that
tilts the LOS plane so `los_rate_el` ≈ 2× `los_rate_az` and the T3's 84° bank puts ~10% of its turn rate into
elevation. Remove the perch, the exploit closes.

⇒ **D3b's positive half must be restated: "a neutral-merge gun kill exists *from a +2000 ft perch*, and not at
co-altitude."** Combined with the already-recorded ±83 ms clock knife-edge and 0/8 on 10/10 perturbed
geometries, instance 3 of the no-BFM synthesis is weaker still. The **admissible** wording is unchanged and
now doubly caveated: *"objective J does not require BFM"*, never an environment fact.

#### Library finding that fell out — the chase cell cannot produce a yo-yo *entry*

`chase{0,30}:high_yoyo` and `:barrel` show **entry 0/8**: `clo_max` is **102 m/s** (chase0) and **76 m/s**
(chase30), both below `CLOSURE_HI = 120`, so the yo-yo entry condition never fires. `chase*:coach` falls
through branch (b) to `LagPursuitTracker` and produces **numerically identical rows** to the non-entering
arms, which confirms it. This is pre-registered outcome **(C)** for the yo-yos and is explicitly **not**
evidence about the detector: the overshoot geometry those primitives exist for **does not occur in this
program's own chase cell against a T3**, so they were never entered rather than flown and missed.

---

### F3 PRE-CHECK — **the `(horizon, rho)` family is CLOSED by arithmetic, and F3 as authored would have made E3 WORSE** (2026-09-06)

The authored F3 box was `horizon 3 → 9` with `rho 0.5 → 1.0` as one joint edit, warm from c96, ~2 h of
learner. **It is not run, and it must not be: the pre-check refutes its premise.** Script
`controllers/rl/_diag_f3_ceiling.py`, output `runs/f3_ceiling.txt`. No GPU, no sim, ~20 s.

**First, two code facts that make F3 a clean single-mechanism edit — both confirmed, and they survive:**
`cfg.horizon` sits under the *MPPI planner* block in `config.py:40`, but `train_parallel.py:4090` states
`cfg.mpc` **is False in every shipped checkpoint** (`--collector-mpc` defaults off) and every eval harness
sets `cfg.mpc = False`. ⇒ **`horizon` drives only the loss window, not the deployed controller.** And
`agent.py:346-348` divides consistency, reward *and* value by `H`, so the (H, ρ) pair is very nearly
scale-neutral (effective magnitude `Σρ^t / H`: 0.583 at the incumbent, 1.0 at H=9/ρ=1.0 ⇒ a 1.71× bump that
would have needed compensating). Both good news — and both irrelevant, because of what follows.

**The pinning claim is RIGHT but the direction is BACKWARDS.** With the `sample()` `[1:]` shift, reward index
`t` reads slot `s+1+t`, the terminal reward sits in the episode's **last** slot `b+T`, and validity requires
no `first` in `s+1..s+H`. Solving: `t ≥ H-1` and `t ≤ H-1` ⇒ **`t = H-1` exactly.** So the terminal is
reachable, always at the far end, in **exactly one** valid window per episode — while the episode offers
`T-H+1` windows. Its share of the reward head's gradient is therefore

    share(H, ρ, T)  =  ρ^(H-1) / [ (T − H + 1) · Σ_{t<H} ρ^t ]

| arm | share | vs incumbent | terminal reward idx |
|---|---|---|---|
| **INCUMBENT** H=3 ρ=0.5 | 9.934e-05 | 1.00× | [2] |
| **F3 as authored** H=9 ρ=1.0 | **7.759e-05** | **0.78×** | [8] |
| H=9 ρ=0.5 (H alone) | 1.367e-06 | **0.01×** | [8] |
| H=3 ρ=1.0 | 2.318e-04 | 2.33× | [2] |
| H=3 ρ=2.0 | 3.974e-04 | 4.00× | [2] |
| H=3 ρ=4.0 | 5.298e-04 | 5.33× | [2] |
| H=1 (degenerate ceiling) | 6.944e-04 | **6.99×** | [0] |

**Verified three ways:** the closed form and a direct enumeration of every valid start agree **to 4
significant figures on all 8 arms**, and a Monte Carlo against the **shipped** `ReplayBuffer._sample_starts`
lands at **0.92–0.99×** of the closed form with the terminal observed at reward index **[2]** (H=3) / **[8]**
(H=9) and nowhere else. The incumbent's 9.93e-05 is also independently corroborated: it sits right in E2a's
measured **0.013%** band for the terminal's share of the loss.

⇒ **The ENTIRE `(H, ρ)` family is capped at 7.00×.** The in-window factor `ρ^(H-1)/Σρ^t` has supremum 1
(ρ → ∞) against the incumbent's 0.1429, and the cap is only approached by the degenerate **H=1**, which
deletes the multi-step consistency loss TD-MPC2 is built on. **E3 needs 43×.** The dominant term is
`(T − H + 1)` — one terminal per ~1440 transitions — and **no in-window reweighting can touch it.**

Two corollaries worth keeping:
- **"H and ρ are one joint lever" was the right warning for the wrong reason.** H=9 at ρ=0.5 is 0.01×, so
  raising H alone is indeed catastrophic — but raising ρ *with* it does not rescue anything, it lands at
  0.78×. The lever does not exist in either coordinate.
- **The direction of the ρ knob is inverted from intuition:** ρ > 1 (up-weight the far end, where the
  terminal always lives) is the only ρ that helps at all, and it buys ≤ 5.33× while making the *dynamics*
  loss dominated by its least accurate step. Not worth a run.

#### ⇒ The replacement box: an explicit terminal importance weight on the supervision

`agent.py:333-334` computes `reward_ce` per horizon position and weights it by `rho` alone. Add a per-sample
terminal weight — `reward_ce` scaled by `(1 + λ·terminated)`, renormalised — and the share moves where it
needs to:

| λ | share | vs incumbent |
|---|---|---|
| 0 (incumbent) | 9.934e-05 | 1.00× |
| 10 | 1.092e-03 | 10.99× |
| **100** | **9.935e-03** | **100.01×** |
| 300 | 2.904e-02 | 292.29× |
| 1000 | 9.046e-02 | 910.54× |

E3's 43× gap sits between λ ≈ 30 and λ ≈ 100, so **λ = 100** is the natural first shot (0.0099% → 0.99% of the
reward head's gradient) with 300 as the second rung. This touches **one term, in one place**: no horizon
change, no planner change, no effective-LR change, no data-distribution change, and it leaves the value head
and the consistency loss alone so the c88/c89 warm-miscalibration failure mode is not in play. It is also
exactly what the E3 verdict already prescribed — *fix the **supervision**, never a reward constant.*

⚠ Pre-register before running: judge on the **out-of-band greedy verdict** (kills, aspect 30 included) with
c96 as the control, **not** on the reward head's own predicted kill−death — that quantity is what λ directly
optimises, so improving it is tautological. The E3 probe (predicted vs true kill−death on c87) is the
*mechanism* check; the greedy verdict is the *result* check. Both, or neither is interpretable.

---

### F4 RESULT — **the T3's post-pass turn is IN-PLANE and it CLIMBS; our dive is a choice, not a constraint** (2026-09-07)

`controllers/rl/_diag_t3_turnplane.py` → `runs/f4/f4_turnplane.txt`. 7 arms × 3 seeds on `merge1500` at
the F2/D3b horizon, so every row is directly comparable to the recorded ones. **Authored to answer a
direct question from the user** — *"does the T3 opponent bank into an in-plane turn after they pass? I
feel like BFM should not cause dives if so"* — and the answer is **yes, and the inference is right, and
stronger than it was put.** `roll_once` is deliberately not reused: it records the *learner's* trace only,
and this question is about the *bandit*.

#### Q_A — in-plane in BOTH senses, and it gains altitude

**Code first (read, `opponent.py`).** After a merge the T3 can only be in `_break` (DEFENSIVE) or
`lead_pursuit` (NEUTRAL/OFFENSIVE). `_break` commands a **fixed** `BREAK_BANK_RAD = 85°` (`:456`) with
**no LOS-plane term**; `lead_pursuit` rolls the lift vector **onto the lead point** (`:338-342`), so its
turn plane *contains us by construction*. **Neither law has an altitude-hold term** (contrast `InnerLoop`,
`:98-107`). The T3 therefore has no mechanism to fly out of plane on purpose, and none to hold altitude.

Measured post-CPA:

| quantity | value | reading |
|---|---|---|
| `\|χ_T3\|`, **all arms, all phases** | **1.7–26.5°** | the honest range; never quote the subset alone |
| — of which NEUTRAL/OFFENSIVE | 1.7–4.6° (valid 74–100%) | lift vector sits *in* the LOS plane |
| — of which the break | 6.7–26.5° | the fixed 85° bank lands near the LOS plane anyway |
| bank `\|φ\|` p50 / p95 | **83.7 / 94.1–105°** | it banks hard |
| turn rate horiz / vert | **8.81–11.5 / 0.71–3.30 °/s** | *these two rows are one arm-subset* |
| vertical share of turn, **all arms** | **2.7–81.3%** | ⚠ the honest range. "77–94% horizontal" was a subset |
| T3 `dalt` vs a level us | **+154 m pre-CPA, +201 m post** | γ peaks +15.0°; apex **+457 m**, net **+355 m** |
| T3 `dalt` vs a turning us | **+763 m post-CPA** | γ p50 +18.3°, max +25.6°; apex **+977 m**, net **+921 m** |

⚠ **FOUR OVERSTATEMENTS IN MY OWN ORIGINAL WRITE-UP, corrected above and here** (found by this box's
own control refuter; the closure is intact, the *numbers* were not):

1. **"ends 457 m higher" / "977 m higher" were zoom APEXES, not end states.** Net window sums are
   **+355 m** and **+921 m**. The direction survives; the figures were the best moment of the climb.
2. **The tight `|χ|` and vertical-share ranges were SUBSETS**, not the measurement. All-arm:
   `|χ_T3|` **1.7–26.5°**, vertical share **2.7–81.3%**. An 81.3% vertical share is not "in-plane
   horizontal" by any reading — the in-plane conclusion rests on the *pursuit law having no
   out-of-plane term* (which is code, not statistics), **not** on these percentiles.
3. **The climb is a PRE-CPA / conversion phenomenon, not a property of the whole engagement.**
   Post-CPA, following a steeply diving learner, the T3 descends **−1176.9 m** and **−1964.4 m**. And
   at **co-altitude** (`d3b@0`) it does not climb at all — γ **−3.3°**. "The merge fight goes up" is
   true of *the conversion after a perched merge*, and false as a general statement.
4. **"c95 −14,807 m" is a POOLED PER-EXCURSION SUM over 63 gated windows** — it is the total of many
   separate descents, not a net altitude change, and it is a member of the same banned family as
   the pooled statistics this document rejects elsewhere. The honest proxy for c95 decking is
   **`hard_deck` 8/12**. **Do not quote −14,807 m.**

⚠ **A PRE-REGISTERED PREDICTION OF MINE WAS REFUTED, and it is in the script header.** I predicted the
break *must* descend, because 85° of bank needs `1/cos 85 = 11.47 g` against a 9 g limiter. **It does not
descend.** The bank-vs-load identity assumes a steady level turn about a *horizontal* axis, which is not
what the T3 flies: it enters the break nose-high, and when the break hands off to `lead_pursuit` — which
has no γ term — the in-plane pull carries that nose-high entry *over the top*. **Do not re-derive the
T3's flight path from its bank angle.**

**⇒ The decision-relevant consequence: the +2000 ft perch is GIVEN AWAY by t ≈ 11 s.** Start Δalt
**+611 m** (= 2000 ft, harness self-check ✓); by t = 11 s on `levelturn80` it is **−122 m — we are below
him** — we sank ~300 m while he climbed ~435 m. Even flying dead level (`straight`) the edge decays
**+611 → +100 m by t = 16 s**. §7's "slight altitude edge for energy" is not a starting condition you
keep; it is one that must be actively defended against his climb.

#### Q_B — the dive is not forced, and the diving arms turn SLOWEST

| arm, post-CPA | `dalt` | horiz turn | `aspect_end` |
|---|---|---|---|
| **`bfm.LeadTurn`** | **+339 m** (+62 pre) | **13.71 °/s** (14.08 pre) | **60.6°** — the only arm < 90° |
| `levelturn80` (new, mine) | −596 m | 8.18 °/s | 134.3° |
| `levelturn60` (new, mine) | −171 m | 2.33 °/s | 171.6° |
| `d3b@2000` (the killer) | −2089 m | 5.47 °/s | 173.4° |
| `d3b@0` | −1940 m | 5.65 °/s | 174.3° |

**The arm that spends the most altitude turns at 40% of the rate of the arm that gains it.** Altitude is
not buying turn rate on this cell. **Mechanism:** at 250–300 m/s a **75°** bank at max g leaves
`5.87·cos 75 = 1.52 g` of vertical lift — above 1 g, so it climbs *while* out-turning. The "altitude is
the turning currency" reading in §9 came from arms banked past ~84°, where `cos φ < 1/nz`. **The lever is
BANK ANGLE, not g.** And **we out-turn the T3** — 13.7–14.1 °/s against its 8.8–11.5 °/s — because it
bleeds to 182–192 m/s converting energy into its climb while we hold ~300 m/s.

#### Caveats, any one of which could carry the conclusion away

- **`levelturn80`'s −596 m is inflated by my own γ-hold loop hunting**: `nz` oscillates 2.31/4.15/5.13/5.40
  at 78.6° bank, median **4.22 g where 5.76 g was required**. It is an upper bound on an 80° level turn's
  cost, **not** an envelope measurement. **Q_B rests on `LeadTurn`, which introduces no new gains.**
- **3 seeds are not 3 independent samples** for any arm that fires nothing — rows are bit-identical, which
  re-confirms F2b's finding that `sim.seed` drives only gun dispersion. Only `d3b@2000` varies (64/99/66
  rounds).
- **`d3b@2000` seeds 11/45 kill AT CPA, so no post-pass window exists** (n = 1). Its POST row is seed 28
  alone — the one that missed. Do not quote it as three samples.
- **`d3b`'s dive is not a failed turn.** It is a deliberate descending slice at **113° bank**; its altitude
  buys *closure and a firing solution*, not turn rate. "d3b dives" and "turning costs altitude" are
  different claims.
- **`LeadTurn` still deals 0.0 HP / 0 rounds / 0 kills on all 3 seeds.** It reaches 227 m and his rear
  quarter (`aspect_end` 60.6°) but with **`ata_end` 138.4°** — a classic overshoot, right position, nose
  not on him. Angles, not lethality; the A2 and F2 closures are intact.

#### What it changes in this document

It reframes §9's *"both checkpoints only know how to turn by spending altitude"* (c95 −744/−801 m,
c87 −2642 m) from a **constraint** into a **choice** — `LeadTurn` turns 13.7–14.1 °/s *while gaining*
339 m, so a high turn rate does not require spending the block. **That is the durable half of F4, and it
is a `bfm.py`/airframe fact, independent of any checkpoint.**

⚠ **The framing I originally wrote here — "the T3 climbs while every trained policy dives, so the SIGN
is wrong" — is WITHDRAWN.** It was built on the apex figures and the pooled −14,807 m, and F5 below
measured the sign contrast directly over whole episodes: **the T3 descends too, by comparable amounts.**
What survives is not a sign asymmetry but an **energy-management failure** (F5), and nothing in the
reward charges meaningfully for descending: `K_CLOSE` −82 and `K_LEAD` −32 are the only large charges,
and the six vertical-coupled terms together are worth only −25 to −40 per episode (F5, Q3).

**Untested implication, worth one box and explicitly NOT a finding:** merge closure is 253–453 m/s (F2),
far above `CLOSURE_HI = 120`, so the overshoot primitives that **never entered** on the chase would
clear *that* clause here, and `LeadTurn`'s overshoot above is exactly what a high yo-yo exists to fix.
⚠ **But clearing the closure clause is not entry.** The chase and the merge fail the yo-yo entry on
**different** clauses — the chase on closure, the merge on another — so "they *would* enter here" is an
**inverted reading of the A2 closure** and is hereby withdrawn as stated. Whether a merge `HighYoYo`
ever enters is unmeasured; that is the box, not the claim.

---

### F5 RESULT — **the dive is real over whole episodes, the SIGN CONTRAST is not, and the mechanism is a zoom-stall** (2026-09-07)

`controllers/rl/_diag_c97_vertical.py` → `runs/c97/c97_vertical.txt`. c96 with **c95 and c87 as D6
controls in the same invocation**, 8 greedy episodes each, `--jitter 0.5`, on the perched merge at the
**full 180 s training horizon** (sep 3000 m, aspect 30°, `opp_heading` 180, `me_dalt_ft` 2000, block
15,000 ft). Authored because F4's conclusion rested on windows of 24 s and on pooled per-excursion sums,
and because three prior reward claims in this document were flagged *"unmeasured, not established"*.

**Two harness traps this box had to code around, both worth remembering.**
`envs.ActionRepeat.step` **sums reward across its 5 ticks but returns only the LAST tick's `info`** — so
accumulating `info["spiral_penalty"]` once per *agent* step samples 1 tick in 5, a ~5× undercount on
every per-tick term and a random 4/5 loss of every PBRS increment. `_refute_alt_tax_episode.py`
accumulates exactly that way; **its reward sums are not comparable to a return.** This probe builds the
env with `action_repeat=1` and does the 5-tick hold in its own loop. Second: the checkpoints are 41-wide
and `OBS_DIM` is now 45, so each is widened at load — verified bit-identical (below), not assumed.

| arm | net `dalt` (median) | climbs | T3 net `dalt` | `hard_deck` | HP dealt |
|---|---|---|---|---|---|
| **c96** (B1 champion) | **−1395 m** | **1/8** | **−532 m** | 1/8 | **0.0 in 8/8** |
| c95 | **−4876 m** | 1/8 | **−4016 m** | **6/8** | **0.0 in 8/8** |
| c87 | **−4880 m** | **0/8** | **−4140 m** | **7/8** | **0.0 in 8/8** |

⚠ **MY PRE-REGISTERED PREDICTION FAILED, and the failure is the finding.** I predicted from the c97
pre-flight (c96 at γ p50 **+26.5°** over 24 s) that c96 would **net-climb**, i.e. that "the policies
dive" was my own artifact. **It does not.** c96 net-descends 1395 m and climbs in 1/8. **The 24 s window
was the opening zoom; the descent is the other 150 s.** ⇒ *"every trained policy dives over a full
episode"* is now **measured and CONFIRMED at the episode level** for the first time — on net `dalt`,
not on a pooled sum.

⚠ **And the SIGN CONTRAST — the actual premise of the c97 box — is REFUTED.** The T3 descends too:
**−532 / −4016 / −4140 m**, tracking each arm's own descent. There is no "he climbs, I dive" state to
expose over an episode; he follows us down. Combined with the co-altitude γ −3.3° above, the T3's climb
is confined to **the conversion after a perched merge**. Anyone reading F4 for a sign asymmetry should
stop here.

**What replaces it (Q2, PREDICTION CONFIRMED) — the failure is ENERGY MANAGEMENT, not vertical sign.**
c96 zooms to γ p95 **+58–68°** on every single episode, and pays for it: **43.7%** of steps below
`V_BAND_CTR` 175 m/s, **21.7%** below `V_CORNER_FLOOR` 150 m/s, `V_min` median **114 m/s** (worst
**98.6**). That is `reward.py:322-327`'s named failure, verbatim: *"the merge zoom-to-stall (airspeed
222→80 m/s, trading all speed for altitude) — the exact hole the T3 exploits (it holds ≥128 m/s in the
vertical scissors while we stall)."* The loop is **zoom → stall → sink → repeat**, and it ends in
`left_bubble`/`timeout` for c96 and in the **deck** for c95/c87.

**Q3 — the reward's vertical tilt, measured at last. It is uniformly ANTI-dive, and uniformly WEAK.**
All six vertical-coupled terms are stored in `info` already signed as their contribution to `reward`
(`K_DECK_FLOOR = K_SPIRAL = −0.02`; the corner floor stores `−pen`), so they are directly summable:

| term | c96 | c95 | c87 |
|---|---|---|---|
| `energy_reward` (K_ENERGY) | −4.79 | −14.03 | −11.63 |
| `rel_energy_reward` (K_REL_ENERGY) | −4.54 | −11.46 | −3.86 |
| `band_reward` (K_BAND) | **−0.00** | −0.00 | **+0.00** |
| `corner_penalty` (K_CORNER) | **−13.31** | −0.82 | −3.09 |
| `deck_penalty` | +0.00 | −3.41 | −8.23 |
| `spiral_penalty` | +0.00 | −11.58 | −11.70 |
| **all six** | **−25.17** | **−39.56** | **−39.62** |
| episode return (median) | −534.8 | −363.6 | −434.1 |

Three things follow, and two of them kill hypotheses I was carrying into c97:

- **"The reward PAYS for diving" is REFUTED.** The supposedly pro-dive pair (`band + corner`) is
  **≤ 0 on every arm** (−13.31 / −0.82 / −3.09). It never pays for descending; it only *charges less*
  than the anti-dive pair does in some arms. The Es PBRS terms are **negative on all three arms**, i.e.
  the reward *is already* charging the energy loss — just weakly.
- **`K_BAND` cannot carry a net vertical tilt over an episode at all**: as a γ=1 telescoping potential
  its episode sum is `K_BAND·(Φ_end − Φ_start)`, and measured it is **−0.00 to +0.00** on all 24
  episodes. Any future argument about the airspeed band must be about *within-episode path shape* under
  the critic's γ = 0.995, never about the reward sum. **A reward-constant edit to `K_BAND` is
  near-inert by construction.**
- **The binding charge differs by arm, which is why one number never described the program.** c96's is
  the **corner floor** (−13.31 — it is slow); c95/c87's is **deck + spiral** (−15.5 / −19.9 — they are
  low). Same reward, two different failures.
- **Scale check, the honest one:** −25 to −40 against returns of −364 to −535. The entire vertical
  pricing is **5–11%** of the return magnitude while the jet gives away the **whole 4.9 km altitude
  block**. The vertical is not mis-signed. It is **barely priced**.

**Also measured, and it is the c97 pre-flight's headline:** widening c96 from 41 to 45 obs via
`_widen_resume_obs` is **bit-identical** — worst `max|Δaction|` = **0.000e+00** over 1152 real merge
steps across `merge@2000` and `merge@0`, encoder pad exactly zero, Adam moments padded rather than reset
(`runs/c97/c97_preflight.txt`). So an obs append on a warm champion is a genuine single-variable edit.

**⇒ What F5 does to the c97 box.** It **removes** the rationale c97 was authored on (the sign contrast)
and **replaces it with a stronger, arm-specific one**: the policy zooms to 60°+ of flight-path angle and
stalls to 114 m/s **while having no observation of its own flight-path angle or vertical rate at all** —
idx 24 `rel_alt` is position-only (3rd-largest first-layer column norm of 41, so height *is* used), and
the horizontal axis by contrast gets both position (idx 14) *and* rate (idx 23). A first-layer
`NormedLinear` cannot form the triple product that would recover vertical rate. **c97 is now an
energy-regulation observability box, not a sign box** — and it must be judged on kills, not on `dalt`.

---

### c97 LAUNCHED — the vertical obs append, PAIRED with a byte-identical 41-obs null control (2026-09-07)

**Both arms launched 2026-09-07, sequentially on the one RTX 5080** (6 collectors each would
oversubscribe if concurrent). ~65 min per arm at the observed 1290 sps ⇒ ~2.2 GPU-hours for the pair.

| arm | `ASTRUM_C97_VERTICAL` | `OBS_DIM` | save path | `_widen_resume_obs` |
|---|---|---|---|---|
| **treatment** | `1` (default) | **45** | `runs/curriculum-97/run.pt` | widens 41→45, pad exactly zero |
| **null control** | `0` | **41** | `runs/curriculum-97-ctl/run.pt` | **not called** (41 == 41) |

Everything else is **c96's exact argv**, copied from `runs/curriculum-96/train.log:68295` and re-emitted
verbatim in both new logs — same ladder, same `--resume runs/curriculum-96/run.pt`, same 5M steps, same
`--workers 6 --rehearse-workers 2`, same `--max-time-s 180 --merge-lateral-m 1400 --me-dalt-ft 2000`,
same `leadturn` override schedule, and `--outcome-reward` on both (the invisible switch).

**Why the control is not optional.** c97 warm-starts from c96, so it gets **5M steps MORE than c96 ever
had**, and the c93 chase autopsy measured that *extra steps alone* can ERODE a warm checkpoint. Without
a 41-obs arm that spends the identical budget, neither a gain nor a null is attributable to the four new
features: a gain could be "more training", a null could be "more training cancelling a real gain". The
control is the same experiment minus the one variable.

**The four features** (appended behind the c87 block so `gun_quality` keeps absolute index 29 and the
fire-override's `GUN_QUALITY_OBS_INDEX − OBS_DIM` self-corrects −12 → −16, verified live at both widths):
idx 41 relative vertical speed `(bandit.vel_z − me.vel_z)/V_REF`, idx 42 `sin γ_me`, idx 43 `sin γ_bandit`,
idx 44 **signed** `bandit.q/RATE_REF` (idx 39's `hypot(q, r)` is left untouched — mutating it would make
this two experiments — so a bandit pulling up and one pushing down stop being bit-identical).

**Switch plumbing, and why it is in three places.** `constants.C97_VERTICAL_OBS` reads
`ASTRUM_C97_VERTICAL`; `geometry.build_observation` appends the four features only when it is set;
`test_dogfight_env.py` gates its slot/sign assertions the same way (**93 passed at 45 AND at 41**). The
var is declared in `docker-compose.yml` as `${ASTRUM_C97_VERTICAL:-1}` because `docker compose run` does
**not** forward arbitrary host env vars — without that line the control would have silently trained the
treatment, which is the `--outcome-reward` invisible-switch failure exactly. And `train_parallel` now
prints an `env: ASTRUM_C97_VERTICAL=… => OBS_DIM …` line next to `argv:`, since an env var leaves no
trace in a launch line. Confirmed in both logs (`obs 45` / `obs 41` in the header, and the treatment
alone logging `[resume] OBS WIDENED 41 -> 45 per frame`).

⚠ **This is NOT a tunable.** The flag exists only so the pair runs from one tree; there is no third value.

**Pre-registered judgment, written before either run finished:**
- Judge on the **out-of-band greedy verdict only** — kills, aspect 30 included, `--jitter 0.5`. Never the
  train log's `kill%` or `ret`. Verify `collected ≈ 5M` in both verdict headers.
- **`dalt` is not the metric.** F5 measured that the vertical is barely priced, so a change in net
  altitude is not evidence either way; the claim under test is that the policy can *regulate energy* well
  enough to convert, not that it flies higher.
- The comparison that counts is **treatment vs control**, not treatment vs c96. If the control also
  moves, the honest reading is "5M more steps", whatever the treatment did.
- Expect the treatment to start **bit-identical** to c96 (zero-padded columns) and have to EARN the four
  features — so an early-training difference between arms is exploration noise, not signal.
- ⚠ A null here does **not** clear the vertical: it would close *this* four-feature append, leaving the
  zoom-stall mechanism itself (γ p95 +58–68°, 43.7% of steps below the band) unaddressed.

---

### c97 VERDICT — **the treatment LOST (4 kills vs its control's 25), and the CONTROL broke the ≥55° wall** (2026-09-07)

Both arms finished at ~5M (`collected` 5,001,664 / 5,000,305, verified in the verdict headers). One
invocation, three checkpoints, identical sweep and seeds: `--modes greedy --aspects 0,30,45,55,60,75
--episodes 20 --jitter 0.5 --separation 700 --max-time-s 60`. Extract in
`runs/c97_paired_verdict.txt`, full log `runs/c97_verdict.log`.

**The anchor validates the whole invocation before any number is read:** c96 re-measured here scores
**45 kills**, with asp0 **8** / asp30 **19** / asp45 **18** — reproducing the recorded B1 table
cell-for-cell. This harness is measuring the same thing the record was written on.

| aspect | c96 (anchor) | **c97-ctl (41 obs, CONTROL)** | **c97 (45 obs, TREATMENT)** |
|---|---|---|---|
| 0 | 8 | 5 | 4 |
| 30 | **19** | 4 | **0** |
| 45 | **18** | 8 | **0** |
| 55 | **0** | **8** | **0** |
| 60 | 0 | 0 | 0 |
| 75 | 0 | 0 | 0 |
| **ALL / 120** | **45** | **25** | **4** |
| mean NET | +42.1 | +39.7 | +16.1 |
| asp55 dealt / rounds | 11.4 / 460 | **64.2 / 98** | 0.0 / 258 |

**1. The four vertical features are a decisive NEGATIVE — and the pre-registration is what makes that
readable.** The treatment scores **4** where its byte-identical 41-obs control scores **25**, same argv,
same budget, same ladder, one variable. Judged against c96 it would have looked like a catastrophe
(45 → 4); judged against its control it is still a clear loss, but a *different sized* one, and the
control is what separates the two effects below.

**2. ⛔ But the treatment never LEARNED to use the features, so c97's actual hypothesis is UNTESTED.**
A pure checkpoint read of `_encoder.0.weight` column norms (no sim):

| | idx41 rel vert speed | idx42 sin γ_me | idx43 sin γ_bandit | idx44 signed q_bandit |
|---|---|---|---|---|
| norm | 0.896 | **0.619** | 0.978 | 1.459 |
| rank of 45 | 42 | **44** | 41 | 39 |

against an inherited-column mean of **4.25** and inherited *minimum* of **0.539** — and for scale, idx24
`rel_alt` is 11.11 (rank 3) and idx23 the horizontal rate is 7.46 (rank 6) in the same matrix. After 5M
steps the four appended columns sit in the bottom five of 45. They started at exactly zero (F5's
preflight) and barely moved. ⇒ **c97 failed at "learn to use the new input", not at "is the new
information useful".** The observability argument F5 authored the box on is neither confirmed nor
refuted; it never got to be the variable. A future retry would have to *force* the features to be read
(separate encoder path, feature dropout on the inherited columns, or training from scratch at 45),
not merely make them available.

**3. ⇒ The honest mechanism is that WIDENING a warm champion is not free, even when the forward pass is
bit-identical.** F5's preflight proved `max|Δaction| = 0.000e+00` over 1152 real merge steps — that is a
statement about the *forward pass at initialisation*, and it was over-read as "a genuine single-variable
edit". Four dead input columns still add parameters, gradient noise and padded Adam moments to the
optimisation, and this pair says that costs kills. ⚠ **n = 1 training run per arm.** Warm continuations
are independently documented to collapse the finish 12 → 0 on their own ([[curriculum-65-settle-and-shoot]]),
so "widening cost 21 kills" is confounded with warm-continuation variance. What is NOT confounded: the
treatment is worse than its own control, and the features are unused.

**4. ✅ THE REAL RESULT IS IN THE CONTROL: the ≥55° wall is BROKEN for the first time in the program —
8 kills / 20 at aspect 55, where c96 re-measured in the same invocation reads 0/20 across 20 episodes.**
This wall has been the declared method ceiling since c36 and survived B1 intact (§9: "0 kills at
55/60/75, all 60 episodes timing out"). It is broken by **5M more steps on the `energy-merge` ladder**,
with the vertical obs OFF. Two reasons to believe it is conversion and not luck: asp55 goes from **11.4
HP on 460 rounds** (c96) to **64.2 HP on 98 rounds** (control) — a **4.7× jump in HP per round** — and
asp45 rises 0 → 8 at the same time. ⚠ **The wall MOVED one rung, it did not vanish:** asp60 and asp75 are
**0 for all three checkpoints**, and the control is the only arm that starts *losing* HP at asp75 (took
11.8 vs c96's 2.6).

**5. ⚠ The control also PAID for it, and the total went DOWN.** 45 → 25 overall: asp30 19 → 4 and asp45
18 → 8 bought asp55 0 → 8. So 5M more steps **redistributed** kills up the aspect hemisphere rather than
adding them, which is the c93 erosion finding ([[curriculum-93-chase-autopsy]]) reproduced on c96 — with
the new detail that what erodes is specifically the *low-aspect* finish. Any future box that wants both
must treat "keep the asp30 finish" as an explicit constraint, not an assumption.

**6. ⛔ The wall was broken with ZERO detectable BFM.** F8 (below) reads c97-ctl at **D1 = 0.00 in all
eight detector configs, D2 = 0.00, D3 = 0.33** on the repaired instrument. The first asp55 kills in
program history contain not one displacement roll and not one yo-yo. This is now the **strongest
policy-side carrier for Direction A** in §10.0 and it is measured post-repair — see F8.

---

### F6 RESULT — **the yo-yo gate IS reachable on the merge (my prediction was half wrong), and flying the yo-yo there makes things WORSE** (2026-09-07)

Script `controllers/rl/_diag_f6_yoyo_gate.py`, output `runs/f6/f6_yoyo_gate.txt`. CPU only, ~10 min, run
alongside the c97 trainer. The question F5 left open and [[t3-turns-in-plane-and-climbs]] made mandatory:
I had written that at the merge, closure 253–453 m/s "far exceeds `CLOSURE_HI = 120`, so the overshoot
primitives that never entered on the chase WOULD enter here" — and that was withdrawn as **inverted**,
because clearing one clause of a three-clause gate is not entry. Whether a merge `HighYoYo` ever *enters*
had never been measured, and it is the prerequisite for any "teach it a yo-yo" box: **a reward or a demo
for a maneuver whose entry gate never opens is a reward for nothing.**

Six arms on merge1500 + a 2000 ft perch, 180 s, three seeds, `jitter = 0`. The two gates under test are
`BFMCoach` branches (b) and (c) verbatim:

    HIGH: aspect < 90°  AND  rng < YOYO_RANGE (900 m)      AND  closure > CLOSURE_HI (120 m/s)
    LOW:  aspect < 60°  AND  rng > YOYO_RANGE_EXIT (1400 m) AND  closure < CLOSURE_LO (40 m/s)

(`aspect` is the angle off the **bandit's tail**, so a head-on merge is ~180°.)

| arm | HIGH joint | HIGH windows | LOW joint | stage-2 entries | end aspect / ata | event | `dalt` | dealt |
|---|---|---|---|---|---|---|---|---|
| `lagpursuit` | **0.053** | 2: [23.4,23.9], [34.5,35.8] s | 0.074 | — | 42 / 31 | `hard_deck` | −4877 m | **0.0** |
| `leadturn` | **0.000** | NEVER | **0.394** | — | 59 / 44 | `hard_deck` | −4879 m | **0.0** |
| `leadturn>highyoyo` | 0.000 | NEVER | 0.394 | **0** (bit-identical to `leadturn`) | 59 / 44 | `hard_deck` | −4879 m | **0.0** |
| `leadturn>lowyoyo` | 0.000 | NEVER | 0.078 | 3, first t=70.9 s, 58.0 s held | **173** / 96 | None | −3403 m | **0.0** |
| `lagpursuit>highyoyo` | 0.022 | 4, first t=23.4 s | 0.160 | 2, first t=23.4 s, 9.4 s held | **165** / 101 | `hard_deck` | −4872 m | **0.0** |
| `lagpursuit>lowyoyo` | 0.016 | 2 | 0.074 | 1, t=10.7 s, 11.9 s held | 79 / 88 | `hard_deck` | −4872 m | **0.0** |

**P1 is HALF REFUTED, and I am reporting the refutation, not the half that survived.** I pre-registered
that the HIGH yo-yo gate *never* opens on either ingress and that the family is "unreachable by
geometry". On the **lag-pursuit** ingress it opens twice — joint 0.053, and the best tick is a clean 3/3
at **aspect 83°, rng 897 m, closure +200 m/s**. So "unreachable" is wrong.

What survives exactly as predicted is the **phase structure**: pre-CPA `aspect < 90` is **0.000 of 178
ticks** while `clo > 120` is 0.972-satisfied; post-CPA the two swap (`aspect < 90` 0.723, `clo > 120`
0.310). The honest statement is therefore **"co-true only in a narrow post-CPA window"**, not "never".

**On the `LeadTurn` arm the HIGH gate is shut for a DIFFERENT reason than I predicted — RANGE.**
`rng < 900` holds on only **0.071** of the episode (0.042 post-CPA) and carries **blame 0.93** among shut
ticks, against aspect's 0.23. A lead turn converts by *staying wide*; it never gets back inside 900 m.
That is a threshold argument, not a geometric impossibility, and it is the one place a threshold probe
would be justified.

**P3 is CONFIRMED, strongly.** The LOW gate is open **39.4%** of the `leadturn` episode (44.6% post-CPA,
3 windows: [70.9,83.0], [93.4,111.0], [136.2,176.1] s). ⇒ **the doctrine answer on this cell is "cut the
corner", not "kill closure"** — consistent with F4's `LeadTurn` ending at aspect 60.6° / ata 138.4°,
which is a low-yo-yo picture, not a high-yo-yo one.

**The decisive new result is what entry BUYS, and the answer is nothing — it costs angles.** The two
`lagpursuit>` arms exist only because the first run refuted P1: once the gate opens, the question stops
being *is it reachable* and becomes *what does it buy*.

- `lagpursuit>highyoyo` diverges from `lagpursuit` at t=23.4 s (post-CPA, so CPA 267 m @ 7.4 s is shared)
  and ends at **aspect 165° / ata 101°** against the pure arm's **42° / 31°**. Same deck, same −4.87 km,
  same 0 rounds. **Flying the HighYoYo exactly when its own gate opened threw away 123° of aspect.**
- `leadturn>highyoyo` is **bit-identical** to `leadturn` and entered stage 2 **zero** times — the
  harness's own control, and it confirms the HIGH row is a real 0.000 rather than a plumbing failure.
- `lagpursuit>lowyoyo` is the only arm that re-closes: CPA moves to 263 m @ **32.7 s**, and its last 66
  ticks are `aspect < 90` **1.000** and `rng < 900` **1.000** simultaneously — the closest any arm came to
  a firing position. It still fires **0 rounds** and deals **0.0 HP**.

⇒ **The yo-yo family on this cell is reachable and unhelpful.** That is a stronger closure than
"unreachable", and it comes with the standing caveat that it is about *this* library at *this* geometry.

⚠ **Two caveats that must travel with every number above.**
1. **All six arms are diving into the ground.** Five hit `hard_deck`; the sixth spends 3.4 km. The
   `lagpursuit*` arms live only ~35 s. So these gate windows occur in a jet on its way to the deck, and
   the reading is "the gate opens during a dive" — not "the gate opens in a healthy fight".
2. **Seeds are bit-identical ⇒ n = 1, not 3.** `build_env` uses `jitter = 0.0` and `sim.seed` only drives
   gun dispersion, so with 0 rounds fired on every arm the three seeds are one sample. The script measures
   and prints this rather than assuming it ([[bfm-learnability-six-disproofs]] mis-filed a 3-seed row as
   3 samples once already).

⚠ A reporting defect in the first run of my own probe, fixed and re-run: `two_stage_ctl` stored a single
`s2_i0` and **overwrote it on every re-entry**, so `leadturn>lowyoyo` printed "stage2 entered t=161.9 s"
when its trajectory had in fact diverged from `leadturn` at **t=70.9 s** (visible as CPA 120 m @ 105 s vs
228 m @ 20.3 s). Every stage-2 entry is now recorded; the table above is from the corrected run. **The
count and the FIRST entry are what make the row interpretable** — a single "last entry" timestamp reads
as "it barely entered at the end" while hiding a 90 s divergence.

---

### F7 RESULT — **the D1 repair works, is OVER-DETERMINED, and still has no positive control** (2026-09-07)

Scripts `controllers/rl/_diag_f7_detector_repair.py` (15 arms × 8 seeds, ~10 min CPU) and
`_diag_f7b_q4_margin.py` (a pure rescore of the saved traces, seconds, no JSBSim). Outputs
`runs/f7/f7_detector_repair.txt`, `runs/f7/f7b_q4_margin.txt`, `runs/f7/traces.npz`.

The four repair steps from §9's F2 spec are landed in `bfm_detect.py` behind four switches
(`D1_REQUIRE_FORWARD_EXIT`, `D1_REQUIRE_NONOPENING`, `D1_ANGLE_CARRIER`, `D1_CLOSURE_OVER_HOLD`), and
an assert in the audit pins the SHIPPED column to the module's live defaults. **`bfm_detect` is imported
only by `test_dogfight_env.py:491` and the `_diag_*` probes — never by the trainer or the reward — so
this touches no run, live or past.**

**TABLE 1, the headline.** Each of r1..r4 differs from `pre` in exactly one switch:

| arm | kills | HP | pre | r1 | r2 | r3 | r4 | r123 | SHIPPED |
|---|---|---|---|---|---|---|---|---|---|
| the ten `chase0:*` / `chase30:*` arms | 0 | 0.0 | **0** | 0 | 0 | **0** | 0 | 0 | **0** |
| `merge:lead_turn` | 0 | 0.0 | **8** | **8** | 0 | 0 | 0 | **0** | **0** |
| `merge:coach` | 0 | 0.0 | **8** | **8** | 0 | 0 | 0 | **0** | **0** |
| `merge:FLYBY(d1exists)` | 0 | 0.0 | **8** | 0 | 0 | 0 | 8 | **0** | **0** |
| `merge:d3b_perch0` | 0 | 0.0 | **8** | 0 | 0 | 0 | 8 | **0** | **0** |
| `merge:d3b_perch2000` | **5** | **99.0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Q1 CONFIRMED.** The four 0.0-HP merge arms go **8/8 → 0/8**, and the `pre` column reproduces F2's
recorded 8/8 exactly, so the harness is scoring the same instrument the record was written on.
**Steps 1–3 alone are sufficient — repair 4 is not needed for the result.**

**Q2 REFUTED — my account of *why* F2 failed was incomplete, and the reason is a real methodological
trap.** I predicted repair 1 alone would be sufficient on all four. It kills only two of them:

- `merge:FLYBY` — its exit has `chi_valid = False`, i.e. the **`chi = 0` sentinel route**, so no valid
  exit exists at all and the event drops.
- `merge:d3b_perch0` — valid exit, but no tick inside `CHI_RETURN_S` has ata ≤ 120°; it never turns back
  in time.
- `merge:lead_turn` and `merge:coach` **survive repair 1** — because **repair 1 is a filter INSIDE the
  exit search, not a rejection.** The loop `continue`s past a disqualified exit and keeps scanning, so on
  a trajectory that turns back toward the bandit it **relocates** the exit later instead of dropping the
  event. TABLE C measures it:

| arm | cfg | exit t | ata_exit | d_ata | d_aspect | closure_exit |
|---|---|---|---|---|---|---|
| `merge:lead_turn` | pre | 9.04 s | 154.4° | +78.7 | −64.0 | −348 |
| `merge:lead_turn` | **r1** | **11.04 s** | **119.8°** | +44.1 | −49.1 | −221 |
| `merge:coach` | pre | 9.29 s | 138.9° | +116.8 | −89.1 | −355 |
| `merge:coach` | **r1** | **10.88 s** | **119.5°** | +97.5 | −72.3 | −239 |

Both relocate to **just inside the bar** — 119.8° and 119.5° against `D1_EXIT_ATA_MAX_DEG = 120`. **The
bar chose the exit.** ⇒ **Adding a constraint inside a search is not monotone in the event count**, and a
threshold placed inside a search behaves as a *selector*, not a filter. This is the second time in two
boxes that a switch I described as one-directional turned out not to be (the first: repair 4, which I had
documented as "the one step that LOOSENS" and which in fact rejects `merge:lead_turn`'s pass outright,
entry +253 m/s but mean −173 over the hold). **Assume nothing about a switch's direction; measure each
one as a marginal against the unmodified instrument.**

**The repair is OVER-DETERMINED. `r2` alone and `r3` alone each kill all four arms** — a one-line change
would have done it. SHIPPED runs all four steps, which is stricter than the evidence requires. The guard
on over-strictness is that the repair's *total* effect on the 15-arm set is exactly **"kill the four merge
passes, change nothing else"**: no configuration removes any event that any other configuration keeps.

**Q3 CONFIRMED, and F7b grounds it mechanically.** `merge:d3b_perch2000` — the **only** arm that kills
(5/8 seeds, 99.0 HP) — reads **0 in every one of the eight configurations, including `pre`**. F7b TABLE A
shows why: its **14 candidate holds die 11 on `hold_s` and 3 on (a) lookback**, both *upstream* of every
repaired clause. ⇒ the repair neither gained nor lost anything on the killer arm, and D1 = 0 there is
**correct**, not a miss: [[merge-kill-reachable-without-bfm]] measured that kill as a head-on face shot
with HCA moving 1.48°, which is genuinely not a displacement roll.

**Q4 CONFIRMED, but F7's own TABLE 3 could not support it and F7b had to.** TABLE 3's `best d_aspect /
best d_ata` columns are computed over `detect_displacement_rolls` **returns** under `pre` — events that
already passed (f) — so on the chase that pool is empty and every cell printed `no cand`, which merely
restates that the `pre` column is 0. The script flagged this in its own NOTE, yet the VERDICT still read
"Q4 CONFIRMED". **What actually carries Q4 is TABLE 1's `r3` column: the full detector with
`carrier='ata'` and nothing else changed, reading 0 on all ten chase arms.** That is a direct measurement.

F7b then supplies the margin the table was meant to give, and it is decisive in the direction I feared
least. Rescoring the saved traces with F2's attribution walk:

- **The recorded chase candidate is reproduced exactly** — four `chase30:*` arms have 8 candidates each,
  all reaching (f) and dying there at **d_aspect −6.5°** (t0 3.00 s, hold 0.542 s, χ peak 73.5°, ata
  5.9° → 21.0°, exit closure +42 m/s). So [[bfm-detector-spec]]'s "the sole chase rejection is (f) at
  −6.5° vs −15°" is vindicated; F7's TABLE 3 was simply reading the wrong pool.
- **On that candidate the ata carrier is STRICTER, not looser: d_ata is +15.2°.** It gains 6.5° of aspect
  while *losing* 15.2° of ata. ⇒ repair 3 does not merely fail to create chase events; it is more
  restrictive on the very candidate that came closest.
- **And the two currencies have OPPOSITE signs on every merge arm**: d_aspect −64 / −89 / −108 / −150
  (all admit) against d_ata **+78.7 / +116.8 / +133.1 / +158.8** (all reject). This is the cleanest
  statement of what F2 diagnosed: the head-on pass buys aspect for free precisely *because* the bandit
  ends up behind you, which is the same fact as losing 79–159° of ata. **Repair 3 discriminates on
  exactly the intended axis.**

⛔ **THE LIMITATION THAT MATTERS: the repaired D1 has NO POSITIVE CONTROL.** F7 removed the false
positives; it did not demonstrate a true positive, because **no arm in the library performs a
displacement roll that converts** — F2 measured D1 = 0/8 on the only arm that kills, and
[[bfm-phase-a-closures]] closed cloning precisely because the stick library cannot fly a yo-yo at any
threshold. So the instrument's specificity is now measured on 15 arms and its **sensitivity is unmeasured
on all of them.** Concretely: a D1 of **zero** is now interpretable ("no roll that gained ata without
opening range"), while a **nonzero** D1 on any cell is a genuinely new observation and must be autopsied
event-by-event — exit ata, exit closure, both currencies — before it is believed or reported.

⚠ **The standing ban is REPLACED, not lifted.** Old form: *"until 1–3 land, do not run any box whose
success criterion is a D1 count on a merge or head-on cell."* Steps 1–3 have landed and are measured.
New form: **a merge/head-on D1 count is now admissible as evidence of ABSENCE and not of presence.** Never
report a nonzero merge D1 without the per-event autopsy; never cite a *pre-repair* merge/head-on D1 count
at all (every one of them was measuring the pass). **D2 has still never fired on anything.**

⚠ **Not re-derived: the POLICY-side chase readings (c95, D2d).** Those need a checkpoint and the GPU,
which c97 is using. They are **predicted** robust — those policies fail at the `hold_s` stage, upstream of
every repaired clause, exactly like `d3b_perch2000` — but that is a **prediction and is labelled as one**
in the script's own verdict block. It is a cheap GPU box once c97's pair is judged.

**The durable side-benefit: 120 traces are now persisted** (`runs/f7/traces.npz`, frozen 17-field order,
with an assert that refuses to re-score if `Sample` changed — *re-fly, do not re-score*). `bfm_detect` was
always a pure function of a per-tick trace and no trace had ever been saved, so every detector question
was paying for JSBSim. F7b is the first dividend: it answered a question F7 could not, in seconds, with no
sim. Every future threshold or clause question on these 15 arms is now free.

---

### F8 RESULT — **the policy-side D1 readings re-derived: my pre-registration was REFUTED, and the record's "c95 = 0.00" was a DROPPED CELL LABEL** (2026-09-07)

`controllers/rl/_diag_f8_policy_d1_reread.py` (4 checkpoints × 3 chase cells × 8 greedy episodes, CPU,
~25 min) and `_diag_f8b_d1_autopsy.py` (a pure rescore of the saved traces, seconds, **no sim**).
Outputs `runs/f8/f8_policy_d1_reread_obs41.txt`, `f8b_d1_autopsy_obs41.txt`,
`f8_policy_d1_reread_obs45.txt`, and `policy_traces_obs41.npz` / `_obs45.npz` (96 + 24 traces in F7's
frozen field order). This closes F7's explicit ⚠: the policy-side readings had **never** been
re-derived under the repaired detector, and F7 labelled their robustness a *prediction*.

**Why it was not merely confirmatory.** F7 measured that the repair is **not monotone in the event
count** — repair 1 filters *inside* the exit search, so it can relocate an exit rather than drop an
event. Relocation runs both ways: a repaired config can land on a tick where (f) passes and **fire where
`pre` did not**. "The repair only tightens, so 0 stays 0" was never an argument.

**✅ The harness is validated against the record before anything is interpreted.** F8's c96 row is
**bit-identical** to the recorded D2d run (`runs/d2d/verdict.txt`): `24 eps | kills 10 deaths 0 decks 0
t/out 14 | dealt 47.2 taken 0.0`, D1 0.00, D3 0.50. Same episodes, same outcomes, same detector output,
different session. So every other row below is a real reading, not a harness difference.

| checkpoint | kills / 24 | dealt | D1 `pre` | D1 SHIPPED | D2 (all cfgs) | D3 (all cfgs) |
|---|---|---|---|---|---|---|
| c95 | 1 | 23.6 | **0.12** | **0.08** | 0.00 | **0.88** |
| c87 | 1 | 25.5 | **0.12** | **0.08** | 0.00 | 0.17 |
| c96 | **10** | 47.2 | **0.00** | **0.00** | 0.00 | 0.50 |
| c97-ctl | **6** | 59.2 | **0.00** | **0.00** | 0.00 | 0.33 |
| c97 (45 obs) | 1 | 17.9 | 0.04 | **0.00** | 0.00 | 0.50 |

**1. ⛔ My pre-registered P1 ("SHIPPED D1 = 0.00 on every checkpoint") is REFUTED, and P3 fired.** c95 and
c87 read **0.08/ep under SHIPPED** — two surviving events each. These are the **first nonzero D1 events
ever recorded on a learned policy** under either instrument.

**2. ⛔ AND THE RECORD WAS BEING MIS-CITED — this is the more important half.** Four places in this doc
and one memory say *"c95's chase is D1 = D2 = D3 = 0.00"*. That number is **c95's asp30 cell only**. This
doc's own cell-resolved table (§9, B1's second half) records **c95 asp55 at D1 = 0.50 and D3 = 2.50 — the
highest of any policy measured** — and F8 reproduces those same 3 events. The summary sentence silently
dropped the cell label and generalised one cell to "the chase". ⇒ **c95 was never a valid carrier for
"kills without BFM"**: it has the *most* detected BFM of any policy and it kills least (1/24 here; 0/6 at
both asp30 and asp55 in the recorded table). It is a **Direction B** carrier. §10.0 is corrected below.
**Lesson: a detector reading without its cell label is not a reading.** Same failure family as
[[curriculum-92-lateral-separation]]'s range-gated window.

**3. The autopsy says the surviving events are BOOKKEEPING — pre-registered outcome A2.** All four sit in
the asp55 wall cell:

| ckpt | ep | t0→t1 | hold | chiPk | nzMin | ata in→ex | d_ata | d_asp | clo in→ex | rngMin | leth in | leth aft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c87 | 6 | 4.25→5.29 | 1.04 s | 128.6° | 3.13 | 17.2→2.2 | **−15.0** | −19.5 | +122→+64 | **256 m** | True | **False** |
| c87 | 7 | 4.25→5.33 | 1.08 s | 126.5° | 3.31 | 17.7→2.6 | **−15.1** | −19.2 | +128→+69 | **268 m** | True | **False** |
| c95 | 4 | 4.17→5.92 | 1.75 s | 131.9° | 3.07 | 28.7→6.7 | −22.0 | −31.8 | +139→+68 | 220 m | **False** | **False** |
| c95 | 4 | 24.25→27.79 | 3.54 s | 94.6° | 3.04 | 42.5→19.4 | −23.1 | −51.8 | +164→+17 | 518 m | **False** | **False** |

- **c87's two events clear the acceptance bar by 0.0° and 0.1°** (`d_ata` −15.0 / −15.1 against
  `ATA_GAIN_DEG` = −15.0). A count that a 0.2° threshold change erases is not a robust count.
- **Both "lethal" flags are `leth in = True, leth aft = False` at rngMin 256 / 268 m** — well inside
  `LETHAL_RANGE_M` = 600. That is the **D4 range artifact** this doc already retracted a D5 claim for:
  at 256 m the range gate is satisfied whatever the lift vector did, and nothing lethal *follows* the
  event. The roll did not create the shot; the jet was already in the envelope while rolling.
- **c95's two events produce zero lethal ticks, and c95's asp55 cell dealt exactly 0.0 HP.**
⇒ report as **"policy-side D1 is nonzero and it still buys nothing"**, never as "the policy flies BFM".

**4. ✅ Non-monotonicity independently REPLICATED on a learned trajectory.** c95 goes `pre` **3 → r1 4**.
F7 found this on scripted arms; it is not an artifact of hand-coded sticks. `r3` (the ata carrier) is the
step that removes an event on both c95 and c87 (3 → 2 each).

**5. ✅ The anti-correlation between D1 and lethality now holds on POLICIES too, post-repair.** The two
checkpoints with nonzero D1 (c95, c87) score **1 kill / 24** each; the two with D1 = 0.00 in *all eight
configs* (c96, c97-ctl) score **10 and 6**. Combined with F2's merge result (8/8 D1 on four arms dealing
0.0 HP, 0/8 on the only killer), the sign is the same on scripted arms and on learned policies, on both
cells. **This is the closest thing the program has to a general law, and it is the reason a D1/D2/D3
count can never be a success criterion.**

**6. ⚠ What this does NOT establish (A3).** Sensitivity is still unmeasured. A2 leaves the repaired D1
with **no positive control** on any trajectory, scripted or learned — the four surviving events are
threshold-grazing chase bookkeeping, not converting displacement rolls. The absence-only rule stands
unchanged. **D2 has still never fired, on anything: 0.00 in all eight configs on all five checkpoints.**

**7. ⚠ D3 is large, config-invariant, and still uninterpretable.** 0.88 / 0.50 / 0.33 / 0.17 per episode
and **identical across all eight configs** (no repair touches D3). LIMIT 3 stands: D3's build clause has
**no magnitude floor**, so these are unpriced. Do not cite a D3 count until it has one.

✅ **Dividend: 120 policy traces are now persisted too** (`runs/f8/policy_traces_obs41.npz`, `_obs45.npz`,
same frozen field order + an `obs_tag` so the two widths cannot be crossed). F8b is already the first
use — the whole autopsy cost seconds. Every future policy-side detector question on these five
checkpoints is free.

---

## 10. Why BFM has not been learned (the structural diagnosis)

### 10.0 ⛔ SUPERSEDING FRAME (2026-09-07): **BFM and the kill are INDEPENDENT on this instrument**

> The rest of §10 was written 2026-09-06 around the claim **"BFM was impossible to DISCOVER, and would
> have been charged for if it happened."** **Both halves are now closed by measurement, and neither
> converted.** D2b/D2c wired the coloured noise into the live `pi` branch and made a 1.5 s coherent roll
> deviation reachable (**0.000 → 0.256**); D2d then ran it and **tied the incumbent** (11 vs 10 kills,
> **p = 1.0000**, D1 = D2 = 0.00 on all three arms) ⇒ **exploration is not the limiter.** Stop-charging is
> closed as a family, blunt (c62) and targeted (D1a) alike. So the 2026-09-06 explanation is *true and
> insufficient*: the two constraints it named were real, both were removed, and nothing changed.
>
> Read 10.1–10.6 as the record of how that was established. Read this subsection for what the
> measurements actually show.

**The finding is a decoupling, and it is now measured in BOTH directions.** The program's implicit premise
was a single causal chain: *learn BFM → gain angles → get kills.* Every instrument says the two ends of
that chain move independently.

**Direction A — kills WITHOUT BFM.**

- **D3b, the merge kill (§9).** The one existence proof of a neutral-merge kill (5/8 seeds, 8 m, 99.0 HP,
  where all three hand-coded BFM arms deal **0.0**) is a **head-on face shot**: the heading-crossing angle
  moves **1.48°** over the firing window. This is a direct kinematic measurement with **no detector in the
  loop**, so it is immune to every instrument caveat in §9 — and it is the strongest single carrier of
  direction A. (⚠ It needs the +2000 ft perch — F2b: perch 0 ⇒ 0/8 kills, **0 rounds fired** — and it is
  knife-edge: ±83 ms of clock shift or 10% of geometry ⇒ 0/8.)
- **B1 / c96 (§9).** The largest gain in program history, **4 → 45 kills from one edit**, bought **zero
  BFM**, and the ≥55° aspect wall is intact. ⇒ the trigger was never the high-aspect limiter, and 45 kills
  needed no maneuver the detector can name.
- ✅ **c97-ctl breaking the ≥55° wall (c97 VERDICT, §9) — the strongest POLICY-side carrier, and the only
  one measured on the repaired instrument.** The first asp55 kills in program history (**8 / 20**, where
  c96 re-measured in the same invocation reads 0/20, at 4.7× the HP per round) come with **D1 = 0.00 in all
  eight detector configs, D2 = 0.00** (F8). A wall the program declared a method ceiling since c36 fell to
  **zero displacement rolls and zero yo-yos.**
- ✅ **c96 on the chase (F8, post-repair).** 10 kills / 24 episodes with **D1 = 0.00 in all eight configs** —
  and this row is *bit-identical* to the recorded D2d run, so it is anchored to the record.

⛔ **RETRACTED 2026-09-07 by F8 — this bullet was in this list and was wrong:** ~~"c95 on the chase: D1 =
D2 = D3 = 0.00 and it still reaches the gun envelope."~~ That 0.00 is **c95's asp30 cell only**; the
cell label had been dropped. §9's own cell-resolved table records **c95 asp55 at D1 = 0.50, D3 = 2.50 —
the most detected BFM of any policy** — and F8 re-measured the same events post-repair (D1 0.12 `pre`,
0.08 SHIPPED). c95 kills **1 / 24**. ⇒ **c95 belongs to Direction B, not Direction A**, and it is listed
there now. A detector reading without its cell label is not a reading.

✅ **F7's caveat on this direction is now DISCHARGED**, and not the way it predicted: the policy-side
readings *were* re-derived (F8), P1 was **refuted** rather than confirmed, and Direction A came out
**stronger anyway** — because the correct carriers (c96, c97-ctl) read D1 = 0.00 in all eight configs,
which is a post-repair measurement rather than an inherited one. **D3b's 1.48° never depended on it.**

**Direction B — BFM WITHOUT kills. This half is entirely detector-independent** (kinematics and HP only),
which is why it is the more secure of the two.

- **F2's four merge arms (§9).** `chiHold` **1.958–2.458 s** at **6.1–7.3 g**, χ peak up to **178°** — real,
  sustained, loaded out-of-plane maneuvering by any definition — dealing **0.0 HP**. One of them
  (`d3b_perch0`) fires **0 rounds** and spends **−2422 m**.
- **F4's `bfm.LeadTurn` (§9).** Turns **13.7–14.1°/s while GAINING 339 m**, and is the **only** arm in the
  library that reaches aspect < 90° — the textbook conversion. **0.0 HP.**
- **F6's six yo-yo arms (§9).** The HIGH yo-yo gate **does** open on a lag-pursuit ingress, and entering it
  at exactly that moment **costs 123° of aspect** (165°/101° vs 42°/31°). `lagpursuit>lowyoyo` holds
  `aspect < 90` **and** `rng < 900` simultaneously for 66 consecutive ticks — the closest any arm came to a
  firing position — and fires **0 rounds**. All six: **0.0 HP.**

⇒ **Neither direction of the implication holds.** BFM is not sufficient for a kill here, and it is not
necessary for one.

**What this changes for planning, concretely.**

1. **A D1/D2/D3 count cannot be a success criterion for a training box.** This was already a standing ban
   ("never report D1/D3 counts as progress"); it now has a *reason* that survives F7's repair. The problem
   is not that the detector is unreliable — post-F7 it is specific on 15 arms — it is that **the quantity
   is not on the causal path to the kill in this environment.** Fixing the instrument does not put it there.
2. **The user's goal and the trainer's objective are different objectives, and no reward constant
   reconciles them.** The stated goal is a fight that *looks like* WVR combat — out-of-plane work,
   displacement rolls, turn-circle geometry. The trainer maximises an outcome that demonstrably does not
   require any of it. Every reward-side attempt to bridge the gap is now closed: blunt outcome-only (c62),
   targeted detector-gated suppression (D1a), the discount pair (c88/c89), lead-null (c74), and the
   `(horizon, ρ)` family (F3 pre-check, closed **by arithmetic**). **Stop looking for the reward term.**
3. **The live hypothesis is that the ENVIRONMENT does not require BFM** — which is a falsifiable claim about
   the opponent and the cell, not another reward edit. The supporting measurements are already in §9: the
   T3 **climbs** and turns **in-plane** (no T3 law has an out-of-plane or an altitude term) and **gives away
   the +2000 ft perch by t ≈ 11 s** (F4); and the merge admits a face shot *from that perch* (D3b) which
   vanishes without it (F2b). A peer that denied the face shot and did not donate its energy would make the
   non-BFM solution unavailable — after which the policy either finds BFM or finds nothing, and **either
   outcome is informative.** ⚠ This is the direction F1 ("expert opponent") was reaching for, and F1 as
   authored is **unsupportable** for a different reason: D3b's knife-edge kill cannot serve as the expert.
4. ⚠ **The opposite reading is NOT excluded, and it is CONFOUNDED — do not adopt either.** Direction B is
   also consistent with *"BFM does not pay in this flight/gun model"* (F6 is direct support: entering the
   yo-yo when its own gate opens made the outcome worse). But **no arm in the library flies BFM well** —
   [[bfm-phase-a-closures]] closed cloning precisely because the stick library **cannot fly a yo-yo at any
   threshold** — so *"BFM doesn't pay"* is inseparable from *"our BFM is bad"* on the present evidence.
   Resolving that confound requires a competently flown maneuver, which is exactly what the library cannot
   supply and what the detector, post-F7, still has **no positive control** for.

---

### 10.1–10.6 (HISTORY: the 2026-09-06 diagnosis, superseded by 10.0)

Added 2026-09-06, prompted by the user asking the right question: *"pulling the trigger in a continuous
observation space is difficult, but learning how to fly BFM should not be."* Correct. **BFM was never
hard to learn here — it was impossible to DISCOVER, and would have been charged for if it happened.**
This section supersedes the lever-by-lever framing of §3/§4 for planning purposes.

Four independent facts, each verified in source, compose into the whole explanation.

### 10.1 The exploration process cannot propose a maneuver — and the fix is DEAD CODE

> ✅ **FIXED 2026-09-06 (D2b/D2c, §9). This section is now HISTORY, not current state.** The seam is
> landed and measured: P(a 1.5 s window holding a mean roll deviation past half deflection) went
> **0.000 → 0.256** on the executed action with `explore_noise=ou --explore-tau-steps 36`. The heading's
> claim is false as of that commit. What remains open is whether reachable discovery *converts* — a
> training run, not a code defect. Read the rest of this section for the diagnosis that motivated it.

`agent.py:27-45`, the comment block above the noise classes, does the arithmetic itself:

> The reference TD-MPC2 draw is i.i.d. N(0,1) per control step, so the exploration SEQUENCE has ~zero
> autocorrelation (measured: `|rho(k)| < 0.04` for every lag on every channel) [...] At 24 Hz that makes
> a 1.5 s (36-step) coordinated deviation — the timescale of a displacement roll / out-of-plane
> reposition — a **~15-sigma event (P ~ 1e-50 per window** at the measured `sigma_ail = 0.21`), i.e.
> UNREACHABLE. White noise can only dither around whatever the prior already does; it cannot propose a
> multi-second maneuver.

⚠ **The numbers in that comment block are wrong, and D2a (§9) replaced them — the conclusion survives on
a corrected basis.** `sigma_ail` on the executed pi path is **0.67–0.90**, not 0.21, and per-step
`|pi−mu|` reaches p90 **0.61–0.91 on a ±1 channel**, so the "15-sigma" framing (which priced a coherent
window as needing many σ of amplitude) is not the mechanism. The mechanism is **N_eff**: measured
**41.2 / 33.9** against the white-noise prediction of exactly **36**, with P(|36-step window mean| > 0.5)
= **0.000 in all 40 channel-views**. That statement is *amplitude-independent* — N_eff is a property of
the correlation, not the scale — which is why the corrected form is strictly stronger than the original:
**raising σ cannot fix this, because σ is already saturating the tanh.** Do not re-quote 0.21 or 1e-50.

`OUNoise` and `PinkNoise` were then implemented carefully to fix exactly this. **They have never run.**

- `act()` `agent.py:178`: `if self.cfg.mpc: return self.plan(...)`. **`cfg.mpc = False` in every
  checkpoint**, so `plan()` is never called during collection.
- `agent.py:181`: `a = mu if eval_mode else pi` — the actor's own i.i.d. Gaussian sample IS the
  exploration.
- The injection `a = a + a_std * self.explore_noise.sample()` is at **`agent.py:245`, inside `plan()`.**
- `agent.py:176` faithfully calls `self.explore_noise.reset()` every episode, on an object that is
  never sampled.

⚠ **§3 row 1 is therefore MIS-FILED and must not be used to close this lever.** It reads "Coloured
exploration — **Dead.** Noise injection lives inside `plan()`, and `cfg.mpc=False` [...] so a noise flag
is a silent no-op." That is not a refutation of the mechanism; it is a statement that **the experiment
never ran.** The correct status is **UNTESTED**. Its secondary argument ("premise was false anyway — the
policy already holds `abs(elev) > 0.5` for up to 29.7 s") is the *command-side* error that D4 made twice
(§9): sustained `elev` is a g command, and says nothing about sustaining a coordinated out-of-plane
ROLL. The load-gated measurement says the opposite — max χ hold 5.40 s, D1 = D2 = 0.00.

**The surgical lever this implies:** wire the existing, already-reviewed noise classes into the path that
actually executes (the `pi` branch at `agent.py:181`), rather than `plan()`. Small, contained, and it
targets the one constraint that makes discovery impossible rather than merely unlikely.

### 10.2 The reward charges for a correct maneuver — but TWO terms, not four, and only on the merge

⚠ **This section was CORRECTED on 2026-09-06.** As first written it listed four simultaneous charges
(aim, closing-range, aspect, energy) by walking a displacement roll through `reward.py` and reasoning
about each term's sign. Two of the four are wrong, and the measurement that says so was already in the
repo: `controllers/rl/_cf_out.txt` (the c93 counterfactual-pull diagnostic) prints the **per-term
episode sums** for three arms on the chase cell — BASE = what the policy chose, PURE/LAG = max-g pull
arms:

| term | BASE | PURE (max-g) | LAG (max-g) | verdict |
|---|---|---|---|---|
| `close` (`K_CLOSE`) | **−82.0** | **−43.3** | **−49.3** | the dominant charge, by 2–3× |
| `lead` (`K_LEAD`) | **−32.4** | **−25.5** | **−26.8** | charge #2 — and it was MISSING from the original table |
| `aspect` (`K_ASPECT`) | −15.2 | **+2.7** | −1.4 | small and **sign-inconsistent** — not a reliable charge |
| `band` / `corner` | −3.9 / −1.8 | −5.6 / −2.5 | −2.8 / −1.1 | small negatives |
| `track` | +4.9 | +6.4 | +5.9 | positive |
| `energy` | **+6.5** | **+7.2** | **+7.4** | **POSITIVE in every arm — not a charge at all** |
| `rel_energy` | — | **+1.8** | **+2.3** | positive |

**The energy row of the original table is simply false**, and [[curriculum-93-chase-autopsy]] already
recorded why in as many words: *"Energy terms are positive in every arm (+6.0 to +7.6) because all arms
gain +2400–3000 m of Es. The 'energy tax forbids the pull' hypothesis is refuted in the strong
direction."* A hold spends 744–2642 m of *altitude* (§9) while Es is conserved as speed — altitude and
Es are not the same currency, and I conflated them.

So the real structure is: **`K_CLOSE` is the charge, `K_LEAD` is the second, and everything else is
noise or pays.** That is a much narrower target than "four terms", and it is why the surviving
stop-charging lever in §6 is `K_LEAD` relief rather than a broad waiver.

⚠ **And the sign is CELL-SPECIFIC — this is the load-bearing correction.** On the chase cell the same
artifact shows the max-g pull arms scoring `Σrend` **−42.7 / −52.8 against BASE's −99.5**, i.e. the pull
is already **+47 to +57 per episode better than what the policy chose, and the policy still declines it.**
On the chase, the reward is *not* the binding constraint. The **−2.94** `HighYoYo` deficit was measured
on the **merge**. So:
- **merge:** the charge story holds; stop-charging is on-target there.
- **chase:** the reward already prefers the pull by a wide margin and the policy refuses ⇒ the binding
  constraint there is §10.1 (discovery) or §10.4 (representation), not the reward.

What survives from the original section, unchanged: `reward.py:157` does state that a
first-pass-then-coast "already nets ≈−48" because "the K_CLOSE/K_AIM PBRS go NEGATIVE on the overshoot";
−2.94 on the merge is *priced*, not free; and **adding a BFM bonus cannot outrun a −82 `close` charge**,
so the instruction stands — do not add a term, stop charging. Only the count and the cell scope change.

### 10.3 The charge is PATH-DEPENDENT — real, but SMALL, and at different lines than first cited

⚠ **Citations corrected 2026-09-06.** These PBRS terms use γ=1 precisely so they telescope — only
endpoints matter, so a wide swing that ends well should cost nothing. A gated potential is not
policy-invariant, so the cancellation breaks whenever a gate opens or closes. That mechanism is real.
But my original citation of `reward.py:332` was wrong: `:332` is the *docstring* describing Φ's
**shape** ("flat at/above corner"), and the implementation four lines down at **`:345` is
`K_BAND·(_band_phi(v) − _band_phi(v_prev))` — an unconditional difference of a state function, which
telescopes exactly.** Shape ≠ gate.

The genuinely non-invariant sites are the three `x if (raw <= 0.0 or gate) else 0.0` forms:

| line | term | gate |
|---|---|---|
| `reward.py:316` | `energy_r` | `engaged = rng ≤ TRACK_RANGE` |
| `reward.py:371` | `rel_energy_r` | same `engaged` gate |
| `reward.py:390` | `lead_r` | `in_zone = aspect ≤ CONVERT_ASPECT_RAD ∧ rng ≤ CONVERT_RANGE_MAX` |

And the magnitude matters: the two `engaged` energy gates are **measured inert** — deleted gain median
0.63 / mean 3.07 undiscounted, ≈ +0.36 discounted against a baseline `|J| ≈ 560` (0.06%), and **exactly
0.000 across the entire chase benchmark**. So path-dependence is not a headline defect; it is
concentrated in `K_LEAD` (`:390`), which §10.2 independently identifies as charge #2. Two facts
pointing at the same term is the closest thing to a converging result this diagnosis has.

### 10.4 Nothing in the stack can represent a multi-second commitment

- `--horizon 3` at 24 Hz ⇒ the MPPI planner looks **125 ms** ahead; the buffer serves `horizon+1`
  slices, so the world model trains on **167 ms** of context. A displacement roll is 5–15 s (120–360
  steps).
- `config.py:69 rho = 0.5`, a per-step loss discount over the horizon (`agent.py:290, 300, 307, 309,
  375`), geometrically downweights depth: `0.5^10 ~ 1e-3`. So **raising `H` does not buy depth** — which
  is what §3 row 2 actually established. ⚠ That row is a refutation of the FIX, not of the MECHANISM,
  and the record has been reading it as the latter. `rho` itself has never been varied.
- Therefore all credit for a maneuver sits in the critic, whose γ=0.995/step gives an 8.33 s horizon —
  barely one maneuver — so a payoff 10 s out arrives worth `0.995^240 ≈ 0.30`, against four charges paid
  now at full weight.
- And that critic trains at **UTD 0.05** (~20 env steps per gradient step). Not a choice: the learner is
  GPU-bound behind 6 collectors.

### 10.5 What this means for the checklist

The program's ordering was backwards: it shaped rewards, then blamed exploration, then the horizon, then
the demos. Every one of those later steps assumed the reward was neutral for BFM. It is −2.94.

Consequences to carry into Phase D:

1. **Two of §3's six "refuted mechanisms" are mis-filed.** Coloured exploration is **untested** (dead
   code path). The short-horizon mechanism is **confirmed**, with its obvious fix blocked by `rho` —
   and `rho` untried. Six negative results in a row on levers that all presuppose a neutral reward is
   itself evidence that the reward was the binding constraint throughout.
2. **Do not test exploration or horizon before fixing the sign — ON THE MERGE.** You cannot measure
   whether correlated exploration discovers BFM while the return for BFM is −2.94: the exploration can
   work perfectly and the return still says don't. That confound invalidates any such result in either
   direction. ⚠ **But §10.2's correction makes this mandate CELL-SPECIFIC.** On the **chase** the
   measured return already favours the max-g pull by **+47 to +57 per episode** and the policy declines
   it anyway — so there is no sign to fix there, and the exploration and representation levers are
   testable on the chase *immediately*, with no reward change and no confound. That is the cheapest
   unconfounded test of §10.1 available, and it was invisible while §10.2 claimed four uniform charges.

   Reference constant for all horizon arithmetic (three Phase-D proposals got this wrong by 5×): one
   control step is **41.7 ms** — `action_repeat = 5` at the 120 Hz sim rate ⇒ 24 Hz policy cadence.
3. ~~The cheapest ordering is therefore: **(a) stop charging**, then **(b) make discovery possible** —
   wire the noise classes into the `pi` branch (`agent.py:181`, `a = mu if eval_mode else pi`).~~

⚠ **BOTH HALVES ARE NOW RESOLVED, and this whole section's ordering is superseded (2026-09-06).**
**(a) stop charging is CLOSED as a family** — blunt (c62, §10.6) and targeted (D1a, §9) alike; there is no
time-local predicate that selects the maneuver over c87's loaded descending break. **(b) is DONE** —
D2b/D2c wired the noise into the live `pi` branch and measured P(1.5 s coherent roll deviation past half
deflection) go **0.000 → 0.256** closed-loop. ⇒ **Consequence 2's mandate is VACUOUS, not merely
cell-specific: there is no sign left to fix first, so nothing is blocked on it.** The live ordering is the
board's, at §6's fired-contingency subsection: **D2d (the run) and D3b (winnability), then the two new
critic defects D5 surfaced** — decoder compression and terminal-transition gradient starvation, which are
the first structurally new levers since B1 and are *not* reward-shaped.

### 10.6 "Go outcome-only" is ALREADY TRIED AND IT DECKED THE CHAMPION (2026-09-06)

Answering 10.5's open question by reading the code rather than launching: **`--outcome-reward` does NOT
replace the dense shaping.** `train_parallel.py:4789` and `reward.py:112-122` — v19b `outcome_mode`
**keeps all five dense pursuit terms** (`K_TRACK/K_SOLUTION/K_CLOSE/K_AIM/K_ASPECT`) and only *adds*
the convert-dwell, `K_ENERGY`, rel-Es, the corner-speed floor and the `CONVERT_CLOSURE_MAX` gate.

And the reason it is a hybrid is recorded at `reward.py:106-110`: **reward-v19's original pure-outcome
form DID drop the five dense terms, and c62 catastrophically collapsed the warm champion — 18 of 24
episodes DECKED at aspect 45/55° by 1.15M steps, against a baseline of 0 losses / 9 kills.** The stated
mechanism: *the dense tracking term is what maintains the ≤55° kill*, and removing it everywhere
(including the rehearsal anchor) left nothing preserving fine control.

⇒ **The blunt half of 10.5(a) is closed.** Going outcome-only is not an untried idea, it is a recorded
catastrophic failure of the same class as [[curriculum-88-discount-disproof]] — an objective shock under
a converged policy. This is the **third** time in this session that a lever reached by reasoning turned
out to be already refuted in the record (the other two: `OBS_DIM 41→44`, more altitude budget).

~~**What survives is the targeted form, and only the A1 detector makes it expressible:** suppress the
geometry terms *during a detected out-of-plane commitment* and nowhere else, leaving the dense finish —
and the rehearsal anchor — fully intact. That is a narrow, time-local waiver of four charges, not an
ablation. It is also the first Phase-D candidate that is not reachable by reasoning alone, which is why
A1 had to be built before B-anything.~~

⚠ **CLOSED 2026-09-06 by D1a (§9, and see the fired contingency in §6).** The targeted form was the last
survivor of stop-charging, and D1a measured its gate predicate before the reward was touched: `inplane`
clears it at least as often as `highyoyo` and **c87 — a loaded, nose-off, descending break that decks —
is the highest-clearing arm of all five**, at all nine (χ, nz) settings, in both duty cycle and per
excursion. **There is no time-local χ∧load predicate that selects the maneuver rather than the failure
mode**, so "suppress the charges during a detected commitment" is not expressible on this instrument even
with the detector built. §10.6's plan of record is **empty**, and stop-charging is closed as a family —
blunt form (c62) and targeted form (D1a) alike.

The lesson to carry forward, because it is the third time this shape has appeared: **the detector is a good
*measuring* instrument and a bad *gating* instrument.** It was built to tell whether a run did BFM after
the fact, on pooled statistics with controls. Wiring the same predicate into the reward asks it to be
right tick-by-tick on a single trajectory, where χ is ill-conditioned near boresight, the load clause is
cleared by any hard pull including a deck-recovery one, and tightening a threshold fragments holds into
blips. Do not put a detector clause on the left-hand side of a reward term again.
