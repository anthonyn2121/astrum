"""Reference scales, ballistics, boundaries and reward magnitudes (design doc §8).

Every value here is a tunable hyperparameter of the *wrapper*, not the sim. They
are collected in one place so a training config can override them wholesale.
"""

import os
from math import radians

# --- observation / action shape ---
# curriculum-87 (2026-09-04): 31 → 41. The obs had been held byte-identical for ~17 consecutive
# runs ([[obs-blind-to-opponent-turn]]) and was the cheapest untried lever; c86's autopsy finally
# named the missing quantity. c86 kills 8/8 at aspect 30 by ARRIVING at 18.6°/s LOS rate and
# firing at 1.1° lead, but at ≥55° entry it closes to 194-232 m and arrives at **59°/s** — flying
# THROUGH the solution instead of settling into it (150-250 m band: 45 rounds, 0% hittable, mean
# lead 28°). LOS rate is what separates those two arrivals and it was not observable, along with
# the opponent's bank/turn-rate (his turn PLANE — the thing "out-of-plane" and "displacement roll"
# are defined against), a SIGNED lead (only the unsigned, saturating `cos(lead_error)` was fed),
# achievability, and relative energy. The 10 appended features are all derived from state the
# adapter already holds, so nothing new is plumbed out of the sim. Appended at the END so
# `gun_quality` keeps absolute frame index 29 and `GUN_QUALITY_OBS_INDEX - OBS_DIM` self-corrects.
#
# curriculum-97 (2026-09-07) appends 4 more: the VERTICAL axis, signed and rate-bearing.
#
# THE ARGUMENT, as revised by F5 (`runs/c97/c97_vertical.txt`) — the original one was refuted and
# is NOT the reason these features exist. F4 was read as "the T3 climbs while we dive, so the SIGN
# is wrong"; F5 measured whole 180 s episodes and the T3 **descends too** (−532/−4016/−4140 m,
# following us down), so there is no sign contrast to expose. What survives is an INFORMATION gap:
#   * `frame_stack = 1` — the policy sees ONE frame, so no rate can be differenced. Every rate has
#     to be handed over explicitly or it does not exist for the net.
#   * The HORIZONTAL axis was handed over as position AND rate: `rng` (idx 14) + `closure` (idx 23).
#     The VERTICAL axis was handed over as position ONLY: `rel_alt` (idx 24) — whose encoder column
#     ranks 3rd of 41, so height IS used. Its rate appears nowhere.
#   * The bandit's climb/descent rate is absent in EVERY form (it is given only as a scalar
#     airspeed, health, alive, and an UNSIGNED |q,r|), and own γ needs products of sinθ/cosθ/α/φ
#     that a first-layer `NormedLinear` cannot form.
#   * And the measured failure is vertical: c96 zooms to γ p95 **+58–68°**, stalls to **114 m/s**
#     (43.7% of steps below the 175 m/s band, 21.7% below the 150 m/s corner floor), then sinks the
#     whole 4.9 km block — 0.0 HP dealt in 24/24 episodes. It is regulating an axis it cannot see
#     the rate of. Same class of gap as curriculum-87's, which is the only lever family that has
#     ever worked here.
# Not a reward box: F5 also measured all six vertical-coupled reward terms as NEGATIVE, totalling
# −25..−40 against returns of −364..−535, so the vertical is barely priced, not mis-signed — and
# `K_BAND`'s episode sum is −0.00 by construction (γ=1 telescoping), so a constant edit is inert.
#
# `--resume` widens the encoder's first layer with ZERO columns (`_widen_resume_obs`), so a warm
# champion starts bit-identical — verified, max|Δaction| = 0.000e+00 over 1152 real merge steps
# (`runs/c97/c97_preflight.txt`) — and must EARN any use of them. See `build_observation`.
#
# ⚠ `ASTRUM_C97_VERTICAL=0` reverts to the 41-float obs. This exists ONLY so the paired NULL
# CONTROL (a c96-recipe continuation at 41 obs) can run from this same tree: c97 warm-starts from
# c96, so it gets 5M steps MORE than c96 did, and [[curriculum-93-chase-autopsy]] measured that
# extra steps ALONE can erode a warm checkpoint. Without the 41-obs arm a null result is
# uninterpretable. Default is ON; the flag is not a tunable.
C97_VERTICAL_OBS = os.environ.get("ASTRUM_C97_VERTICAL", "1").lower() not in ("0", "false", "no")
OBS_DIM = 45 if C97_VERTICAL_OBS else 41
ACTION_DIM = 5  # [aileron, elevator, rudder, throttle, trigger]

# --- normalization reference scales (obs → ~[-1, 1]) ---
V_REF = 300.0  # m/s; airspeed / closure scale (f16 WVR regime)
ALT_REF = 6000.0  # m; altitude / relative-altitude scale
RATE_REF = 3.3  # rad/s; body-rate scale (~188°/s max f16 roll rate, §2.4)
ALPHA_REF = radians(30.0)  # rad; the f16 AoA limiter (§2.4)
BETA_REF = radians(20.0)  # rad; sideslip scale
NZ_SCALE = 8.0  # (Nz - 1) / NZ_SCALE → ~1 at the 9 g cap
R_MAX = 5000.0  # m; range normalization (beyond WEZ, within the merge)
HP_MAX = 100.0
# curriculum-87 obs-append scales.
G_MPS2 = 9.80665  # standard gravity, for specific energy Es = alt + V²/2g
LOS_RATE_REF = radians(60.0)  # rad/s; LOS-rotation-rate scale. 60°/s is the MEASURED fly-through
#     rate at 150-250 m (0% hittable); the killing arrival runs ~18°/s. So ~0.3 = a tracking
#     arrival and ~1.0 = a snapshot crossing — the discriminator sits mid-range, not at a tail.
MISS_REF = 40.0  # m; predicted-miss scale for `lay_miss_m`. 5× the 8 m hit sphere, so a genuine
#     solution lands ~0.2 and the 28°-lead close passes saturate — the feature has to resolve the
#     NEAR-solution band (measured 250-600 m is 52-60% hittable) rather than the hopeless tail.
ES_REF = 3000.0  # m; specific-energy-difference scale. c86's frontier cells sit -473..-528 m at
#     closest approach vs -294 m in the killing cell, so the interesting band is a few hundred
#     metres; 3000 keeps a full-blown energy collapse inside ~1.
AMMO_MAX = 511.0  # M61A1 magazine (§3.3)

# --- gun ballistics for the WEZ / lead feature (§3.3; approximate, drag-free) ---
MUZZLE_MPS = 1030.0  # muzzle velocity
R_EFF = 1200.0  # effective gun range (m)
LEAD_SIGMA = radians(3.0)  # rad; tracking-error scale for the gun-solution Gaussian

# --- G1 (curriculum-78, safe-core): RANGE-DEPENDENT HONEST lead tolerance for the REWARD-only
#     gun-solution quality (`gun_quality_honest`). The fixed LEAD_SIGMA=3° Gaussian above is
#     ~3-4× LOOSER than the true 20mm hit threshold at range (documented calibration mismatch,
#     [[curriculum-76-77-bleed-corner]]): the target's 8 m kill sphere + the 8 mrad dispersion
#     cone subtend only arctan((8 + σ_disp·rng)/rng) — ~1.39° @494 m, ~0.65° @1200 m — so a 3°
#     "near solution" at the merge crossing scores gun_quality≈0.36 yet lands 0 hits (494·sin3°
#     ≈ 26 m ≫ 8 m). `sigma_track(rng)` (geometry.py) TIGHTENS the tolerance with range but is
#     CLAMPED at LEAD_SIGMA_CAP (= the old 3°), so at the close chase-finish (≤~180 m, where the
#     8 m sphere subtends ≥3°) the honest signal is IDENTICAL to today — the finish is preserved;
#     only un-hittable long-range shots lose their false credit. This drives the REWARD math only
#     (K_SOLUTION / K_AIM under outcome_mode); the OBSERVED gun_quality feature (obs idx 29) is
#     LEFT on the fixed LEAD_SIGMA so the warm net's input distribution is byte-identical.
HIT_RADIUS_M = 8.0          # = sim CombatParams.hit_radius (resources.rs): the target's kill sphere (m)
DISP_SIGMA = 0.008          # = sim Gun.dispersion_rad (components.rs): 8 mrad 1σ round-to-round cone (rad)
DISP_MARGIN = 1.0           # dispersion σ's folded into the hittable angular radius (primary knob)
LEAD_SIGMA_FLOOR = radians(0.30)  # rad; tightest honest tolerance (long range floor)
LEAD_SIGMA_CAP = radians(3.0)     # rad; = LEAD_SIGMA → ≤~180 m the honest signal is byte-identical to today

# --- B1: the MEASURED per-round hit probability vs range (the real WEZ) ---
# `sigma_track` above made the ANGULAR tolerance honest. This makes the RANGE weighting honest.
# Until now `range_factor` was `(R_EFF − rng)/R_EFF` — a linear ramp that pays 0.42 at 700 m and
# 0.25 at 900 m, where the gun in fact cannot hit anything. The numbers below are Monte-Carlo
# measured through the SIM's own ballistics (`controllers/rl/_diag_phit_range.py`, whose integrator
# is pinned tick-exactly to `ballistics.rs` by the golden trajectory in
# `crates/astrum-sim/tests/ballistics.rs::golden_trajectory_pins_the_python_mirror`), so this is the
# one range curve in the repo that is not a guess.
#
# ⚠ MEASUREMENT CONDITIONS — the table is p_hit(range) and therefore hardcodes everything else:
# 5000 m altitude, shooter 230 m/s / bandit 200 m/s, pure tail chase, bandit pulling **5 g**, and the
# COMPUTABLE lay (constant-velocity `gunlay.intercept`, i.e. what a real FCS can actually solve —
# not the clairvoyant lay fitted to the arc the target ends up flying). The 5 g choice is the
# conservative-but-relevant one: the fixed T3 rides its AoA limit in the break
# ([[aggressive-t3-break]]), and g-sensitivity is large — at 700 m a 1 g bandit measures 0.594 vs
# 0.060 here. Against a NON-maneuvering target this curve therefore UNDER-credits a long shot by up
# to ~10×. That is the deliberate trade: the policy's failure mode has always been firing at a
# median 1519 m, and there is no non-maneuvering opponent in this curriculum.
#
# The decay is target maneuver during time of flight, measured not assumed: with the clairvoyant lay
# the same 5 g case still reads 0.675 at 600 m, so the collapse to 0.202 is the bandit moving while
# the round is airborne, not dispersion and not drag.
PHIT_RANGE_M = (200.0, 300.0, 395.0, 440.0, 500.0, 600.0, 700.0, 900.0, 1200.0)
PHIT_PER_ROUND = (1.000, 0.983, 0.834, 0.701, 0.484, 0.202, 0.060, 0.002, 0.000)
# ⚠ Below 200 m the measured curve SATURATES at 1.0, so `range_factor` is flat there and the
# `K_AIM` aim-PBRS loses its range gradient inside 200 m (closing 200→150 m used to pay +2.5, now
# 0). Kept flat deliberately rather than inventing a slope: p_hit really is ~1.0 there, the angular
# gradient (the `track` Gaussian) is untouched, `K_CLOSE`'s potential is already clipped at
# `R_CLOSE_MIN` = 250 m, and [[curriculum-93-chase-autopsy]] found range is NOT the chase limiter.
# What this DOES change is the allocation of the (telescoping, episode-bounded) aim payment: the
# 1200→600 m leg drops 30 → 12.1 and the 600→150 m leg rises 22.5 → 47.9. That reallocation — pay
# for entering the lethal envelope, not for closing to standoff — IS the intended lever.

# The lethal envelope. `in_solution` (reward.py) and the D5 `lethal` tick (bfm_detect.py) BOTH gate
# on this, from here, so the reward and the BFM instrument can never drift apart about what a
# killing shot is. `test_gunlay.py`'s independent drag-corrected curve puts p_hit < 0.05 at 600 m
# against an 8 g break, so 600 m is if anything the generous edge of the envelope.
LETHAL_RANGE_M = 600.0
# ⚠ `R_EFF` (1200 m) MUST NOT be collapsed onto this. It is the only thing `in_range` reads, and
# `in_range` is what the SCRIPTED T3's own trigger fires on (`opponent.py:327`). Moving it would
# stop the benchmark shooting in the 600-1200 m band and void every score measured against it — the
# same class of silent-benchmark-change as [[t3-break-ungated-pull]]. Pinned by
# `test_dogfight_env.py::test_r_eff_is_the_opponent_benchmark_not_the_lethal_envelope`.

# --- episode boundaries the wrapper enforces (§8.4) ---
DECK_AGL = 300.0  # m; hard deck (~1000 ft AGL) — recovery room above real terrain
R_BUBBLE = 15000.0  # m; horizontal engagement-bubble radius from arena center
CEILING_AGL = 15000.0  # m; engagement-bubble ceiling above ground

# --- Tier-2 scripted opponent: defensive maneuvers (§8.5) ---
R_DETECT = 4000.0  # m; range within which a rear-hemisphere attacker triggers a break
REAR_HEMI_RAD = radians(90.0)  # ATA(me→attacker) beyond this ⇒ attacker is aft of my 3-9 line
V_CORNER_MPS = 175.0  # m/s (~340 kts); corner speed — cap sustained pull to hold it (§8.5)
ALPHA_SOFT_RAD = radians(22.0)  # self-imposed AoA soft cap (below the FLCS 30° hard limiter)
BREAK_BANK_RAD = radians(85.0)  # commanded bank for a hard defensive break turn — lift vector
# nearly ONTO the threat (a max-performance break), so the turn is in-plane/nose-low, not a wide
# energy-preserving climbing turn (which had a ~3x larger radius than a real hard break).
BREAK_REVERSE_DEADBAND_RAD = radians(5.0)  # azimuth the threat must cross BEYOND the nose before a
# break reverses its turn direction. Without it, `Tier3Opponent._break` recomputed its roll sign from
# `ata_az` every tick and chattered whenever the attacker sat near the nose. See the roll-then-pull
# gate in `_break` for why an ungated reversal was catastrophic.
BREAK_SWEEP_RAD = radians(150.0)  # heading swept before a break turn recovers
SPLITS_SWEEP_RAD = radians(165.0)  # ground-course reversal before a Split-S recovers
MANEUVER_MAX_S = 12.0  # hard safety cap on any single maneuver's duration
RECOVER_S = 1.0  # wings-level rollout after a maneuver before returning to hold
MANEUVER_COOLDOWN_S = 3.0  # min dwell in level-hold before a maneuver may retrigger (anti-chatter)

# --- Tier-3 FSM tactician: BFM thresholds + WEZ trigger (§8.5) ---
# All transitions are dead-banded (separate enter/exit thresholds) and dwell-gated.
FIRE_LEAD_RAD = radians(4.0)  # gun-solution cone: fire when lead error < this and in range
THREAT_CONE_RAD = radians(45.0)  # attacker's nose-to-me angle below which I'm threatened
THREAT_CONE_EXIT_RAD = radians(65.0)  # ... and above which the threat is considered cleared
THREAT_RANGE = 2500.0  # m; attacker within this (and nose-on) ⇒ defensive
THREAT_RANGE_EXIT = 3500.0  # m; beyond this ⇒ threat cleared
OFF_ATA_ENTER_RAD = radians(45.0)  # my-nose-to-target below this (+ close) ⇒ go offensive
OFF_ATA_EXIT_RAD = radians(70.0)  # ... above this ⇒ drop back to neutral
OFF_RANGE = 3000.0  # m; commit-to-offensive range
OFF_RANGE_EXIT = 4500.0  # m; disengage range
CLOSURE_HI = 120.0  # m/s; closing faster than this up close ⇒ overshoot risk ⇒ high yo-yo
CLOSURE_LO = 40.0  # m/s; closure bled off ⇒ back to offensive
YOYO_RANGE = 900.0  # m; overshoot only matters this close
YOYO_RANGE_EXIT = 1400.0  # m; range reopened ⇒ resume pursuit
V_ENERGY_LOW = 105.0  # m/s; airspeed floor in a break ⇒ energy recovery (lowered from 120 so an
# aggressive break commits deeper — bleeds down toward corner and expends energy — before bailing)
V_ENERGY_OK = 175.0  # m/s (~corner); recovered ⇒ re-engage
V_PURSUIT = 200.0  # m/s; target speed the pursuit throttle law holds
# --- exploration-only PursuitExpert closure controller ---
# The expert holds its speed at the target's speed + a *commanded closure* that tapers with
# range, so it blazes in from standoff yet settles into a stable guns-trail instead of
# overshooting (a fixed-margin chase closes fast then blows past, unable to re-solve). The
# commanded closure is K_CHASE_RANGE·(rng − R_CHASE_TRAIL), clamped to [0, V_CHASE_MARGIN]:
# = V_CHASE_MARGIN far out (max overtake) → 0 at R_CHASE_TRAIL (co-speed trail at guns range).
# (V_PURSUIT alone is < cruise, so the bare Tier-3 hold would decelerate below a fleeing target
# and never close — Tier-3's own V_PURSUIT hold is unchanged.) R_CHASE_TRAIL is held moderately
# deep (~300 m) so the *converging* pass — whose geometry yields the brief sub-degree tracking
# windows the fire gate needs — runs long enough to gun the kill, without pressing so fast up the
# tailpipe that the (200-m/s-tuned) tracker porpoises and the aim blows up short of a kill.
V_CHASE_MARGIN = 60.0  # m/s; max overtake (closure) commanded when far out
K_CHASE_RANGE = 0.06  # 1/s; closure per metre of range beyond the trail (proportional keeper)
R_CHASE_TRAIL = 300.0  # m; range the commanded closure tapers to zero (stable guns-trail)
# The expert fires only when its predicted lateral miss (rng·sin(lead_error)) is within this —
# i.e. the aim point falls inside the target's ~8 m hit sphere, where rounds actually connect.
# The bare WEZ lead cone (4°) legally fires at long range too, but there the 8 mrad dispersion
# footprint (and any lead error) dwarfs the target, so the whole 511-round magazine misses. This
# gate concentrates ammo on close shots that hit (see sim CombatParams: hit_radius 8 m).
# NOTE (reward-v14): this constant is the SCRIPTED fliers' hard fire TRIGGER (PursuitExpert
# opponent.py:_fire). The reward's WEZ/waste split is DECOUPLED to WEZ_HIT_M below — tightening the
# reward gate must NOT shrink the expert's trigger (6 m here made the expert time out gunning even a
# straight Tier-1 dummy), and the T3 bandit fires off the 4° lead cone (FIRE_LEAD_RAD), not this.
GUN_FIRE_MISS_M = 10.0
# reward-v14 (2026-08-29): the LEARNER's reward WEZ/waste split, decoupled from the expert trigger
# above. c36 verdicts pinned the wall at aspect ≥55° to TERMINAL-KILL PRECISION, not time: the policy
# chases the T3 down to ~2 HP then sprays its WHOLE 511-round mag (0.19 HP/round vs 0.9 at 45°) and
# never lands the kill (a 98-HP ceiling at ANY clock). The old 10 m gate was 2 m LOOSER than the sim's
# real 8 m hit_radius, so grazing shots that physically cannot connect still scored "in-solution" and
# were PAID +R_FIRE_WEZ (a full out-of-solution mag-dump would cost ≈−257 via the waste ramp — the
# policy would never do it, so the spray must be scoring in-WEZ). 6 m sits INSIDE the 8 m sphere and
# TIGHTER still against a jinking target (which moves between trigger-pull and bullet-arrival), so only
# center-mass aim that actually connects earns the bonus; grazing sprays flip to the steep waste ramp.
# As a REWARD gate (not a hard fire ban) this is low-risk: the learner may still fire a looser shot, it
# just forfeits the bonus and eats waste.
# reward-v14b (2026-08-29): 6 m OVER-tightened — c37 REGRESSED the proven 45° kill (12/12 @700k → 0/12
# @1.4M: never-fire creep, fired only 99 rd and left the bandit at ~4 HP to time out) while NOT curbing
# the 60° spray (315 rd). A single distance threshold is a blunt tool: "predicted lateral miss < gate"
# does NOT imply a hit on a jinking target, so 6 m penalised rounds that ACTUALLY land the 45° finishing
# burst (its terminal shots sit in the 6–8 m band) yet still paid the 60° graze. Corrected to 8.0 = the
# sim's exact hit_radius: pay the WEZ bonus iff the shot can PHYSICALLY connect, removing only the old
# 10 m "pays for impossible shots" loophole without demanding tighter-than-physics precision that kills
# the finishing burst. (Also note: the 511-rd mag-dump was a 120s-clock artifact — at the 60s clock the
# policy already fires a disciplined ~182 rd, so fire-discipline tightening has little upside here; if
# 8 m still cannot crack the ≥55° terminal kill, the lever is frontier-concentration, not the fire gate.)
WEZ_HIT_M = 8.0
PURSUE_ROLL_DEADBAND = radians(5.0)  # stop rolling once the target is this near boresight
K_PURSUE_ROLL = 2.0  # lift-vector-on-target roll gain
K_PURSUE_PULL = 1.5  # pull gain (× off-boresight angle) once rolled on
FSM_MIN_DWELL_S = 0.4  # min time in a state before a non-survival transition (anti-chatter)

# --- reward magnitudes (§8.3; reward-v2 rebalance 2026-08-26) ---
# reward-v1 (R_KILL +100 / R_DEATH −100 / no standing tracking payoff) plateaued the
# learner at a passive "survive-to-timeout" draw: the geometry shaping was
# potential-based (telescoped to ~0 over an episode), the kill was a rare sparse event,
# and the death/deck penalties made any imperfect offense EV-negative against a competent
# bot — so gradient descent parked at the ≈−7.2 pure-time-penalty draw. reward-v2 makes
# tracking pay densely, disciplines the trigger, and softens the risk of committing.
R_KILL = 150.0  # terminal: bandit destroyed (raised — reward committing to the kill)
R_DEATH = -60.0  # terminal: learner destroyed (softened — don't punish improving offense into passivity)
R_TRADE = -25.0  # terminal: mutual kill (both die same tick) — worse than a draw, better than a clean death,
#                  so the agent neither seeks a suicidal trade nor fears taking the bandit down with it
R_DECK = -200.0  # terminal: hard-deck bust (unchanged — self-inflicted; keep it the single worst outcome)
R_BUBBLE_EXIT = -50.0  # truncate: left the engagement bubble
# reward-v7 (2026-08-27): TERMINAL penalty for an UNFINISHED engagement — a clock timeout with
# no kill/death and inside the bubble. GROUND-TRUTH greedy eval of the reward-v6 policy came back
# 0/20 kills at BOTH 700 m (44 HP dealt) AND point-blank 400 m (30 HP): the deterministic policy
# closes, chips, and coasts to a draw even when spawned inside gun range — so the bottleneck is not
# range/aim but that its own chip-and-draw behavior is POSITIVELY rewarded (+45..+71 return: banked
# closing-PBRS + tracking cap + partial damage), giving gradient descent no reason to improve it
# toward the kill the exploration/override already finds (Q≈+200). This makes a no-kill draw a
# FAILURE, not a comfortable outcome: it lowers the value of every draw-bound trajectory, widening
# the kill-vs-draw gap the policy must climb. It is applied in DogfightGymEnv.step (it needs the
# sim's clock-truncation flag), NOT in the stateless reward.compute. The closing-PBRS + tracking
# terms are SHARED by a kill and a draw (both end close & nose-on) so they cancel in the decision —
# the penalty is what tips it. Wired PER-STAGE via Stage.timeout_penalty: this magnitude on the
# offensive/kill rungs, but ZERO on the T3 point-blank *survive* rungs, where surviving to the clock
# IS the win and a penalty would fight the objective. At T1/T2 the bandit never fires, so no death is
# possible and this cannot induce a suicidal trade; the only competing outcomes are kill/deck/bubble.
# curriculum-34: lowered −80 → −40 to stay MILDER than |R_DEATH| (−60). The −80 magnitude was set for
# the T1/T2 kill rungs where the bandit never fires (death impossible → no suicide risk, per above), but
# the c34 kill-gated ladder puts metric="kill" on live-firing T3 rungs; there a timeout worse than a
# death (−80 < −60) would make a suicidal merge less-bad than a draw. −40 still makes a chip-draw clearly
# negative-EV (and a draw is a gate-LOSS on a kill rung regardless), while a death stays strictly worse.
# curriculum-36 (reward-v13): −40 → −50. With PASSIVE_TIMEOUT_MULT flattened to 1.0, EVERY no-kill clock
# draw is now this single value (no engaged/passive split); −50 makes an engaged chip-draw distinctly worse
# than the old −40 (so the policy can't settle for "connect a bit and time out") while staying milder than
# |R_DEATH| (−60) so death remains the worse non-deck outcome (no suicide incentive). Dense damage still
# credits a real chase (DMG_PER_HP), so engagement is rewarded on the WAY IN — only the terminal draw is flat.
R_TIMEOUT = -50.0  # terminal: clock ran out with no kill (kill-metric stages only; 0 on survive rungs)
DMG_PER_HP = 0.5  # dense: ±0.5 per HP dealt / taken (a 2-HP hit = +1.0)
TIME_PENALTY = -0.001  # dense: per-step time cost
GAMMA = 0.99  # discount (unused by reward-v2's dense terms; kept for the buffer/agent)

# --- dense tracking reward (reward-v2): a NON-telescoping standing incentive to point
#     the nose at the bandit, close the range, and hold a gun solution. Replaces the old
#     potential-based K_GEOM·(γΦ′−Φ) shaping (policy-invariant → tracking earned nothing
#     cumulatively). Calibrated per SIM TICK (~7200/episode; ActionRepeat sums them).
TRACK_RANGE = 3000.0  # m; pursuit is rewarded within this range (wider than the R_EFF gun range)
K_TRACK = 0.01  # per-tick weight on nose-on × range-closeness (merge / pursuit shaping)
K_SOLUTION = 0.03  # per-tick weight on in-WEZ gun-solution quality (the pre-fire "in the money" signal)
K_GEOM = 0.08  # LEGACY (reward-v1 PBRS scale) — no longer used by reward.compute; kept for reference

# --- reward-v4: per-EPISODE cap on the cumulative tracking sub-total. reward-v2/v3 left
#     tracking UNBOUNDED: (K_TRACK+K_SOLUTION)·7200 ≈ 288 ≈ 2·R_KILL, so a full-episode
#     nose-on track-draw out-earned a kill — and a kill ENDS the episode, cutting off the
#     tracking stream — which reincarnated the passive-draw trap at a higher reward level
#     (curriculum-03: ret climbed 40→110 with kill-rate stuck at ~2% for 300k updates).
#     Clamping cumulative tracking at ≈R_KILL/3 makes a kill (banked tracking + ~50 damage
#     + 150) strictly and hugely dominate any track-draw (≈TRACK_BUDGET − time penalty),
#     while keeping the full-rate tracking gradient for the first ~9 s of good pursuit each
#     episode (strong early "learn to point" signal, then only damage→kill pays). The fire
#     bonus (ammo-limited to ≤511·R_FIRE_WEZ ≈ 25) and damage (100 HP = a kill) self-cap, so
#     they are NOT clamped. Enforced in DogfightGymEnv.step (reward.compute stays stateless).
#
# reward-v5 (2026-08-27): TRACK_BUDGET 50 → 15. curriculum-07 mastered stage 0 (T1@400m,
#     70% kills, gate) but then STALLED at stage 1 (T1@700m): ~2M steps / ~1700 episodes with
#     early guided-fire help, kill-rate flat at 0-6% while ret(20) sat *steady* at ~+70 — the
#     textbook signature of a shaping-reward local optimum, not a skill floor (a skill floor
#     gives low, noisy returns; this was a high, STABLE plateau). Diagnosis: at 700m the policy
#     can hold approximate aim (banks the +50 tracking cap) and chip partial damage (+30-40,
#     uncapped) for a no-kill farm of ~+70-90 — competitive with a +150 kill once the
#     overshoot/bubble-exit risk of committing to the final approach is priced in, because the
#     tracking Gaussian pays for aim WIDER than the hit cone (calibrated for point-blank). At
#     400m the kill is nearly free so it happens; at 700m the safe farm wins. Cutting the
#     bankable tracking total to ≈R_KILL/10 makes a kill (~215) dominate any farm (~15 + chip)
#     4-14×, forcing the policy off the farm and onto the damage→kill chain — WITHOUT touching
#     the per-tick tracking *gradient* (still full-rate "close and point your nose" for the
#     first ~6 s of every episode, so stage 0's point-blank mastery does not regress). Damage
#     credit is untouched: it is earned only on real hits, so it is honest progress, not a farm.
TRACK_BUDGET = 15.0  # ≈ R_KILL/10; max cumulative tracking reward creditable per episode (reward-v5)

# --- reward-v6 (2026-08-27): potential-based CLOSING-RANGE shaping. Ground-truth eval of the
#     reward-v5 policy at 700 m (greedy, no-MPC, 20 eps) came back 0 kills / 0 deaths / 20
#     TIMEOUTS, 8 HP dealt: the deterministic policy learned to park nose-on at STANDOFF, bank
#     the (now-15) tracking cap, and coast to a draw — the ~30% "kills" in the training log were
#     exploration-noise, not a learned skill. Root cause: even capped, the tracking term is
#     FULLY earnable at 700 m without closing (nose-on at 700 m accrues the whole budget in ~6 s),
#     so once the cap is banked there is ZERO gradient rewarding getting closer — v5 removed the
#     very "close the distance" pull. This adds a potential-based term F = Φ(rng′) − Φ(rng) with
#     Φ(r) = −clip(r, R_CLOSE_MIN, TRACK_RANGE)/TRACK_RANGE, which pays to REDUCE range (down to
#     gun range) and costs to open it. It is POTENTIAL-BASED (telescoping) so it (a) cannot be
#     farmed — sum over an episode is just Φ(end)−Φ(start), independent of path — and (b) does
#     not change the optimal policy, only makes the sparse close→convert path learnable. CRUCIAL:
#     the discount in F is 1.0, NOT γ. A γ<1 PBRS on a NEGATIVE potential yields a standing bonus
#     γΦ−Φ = Φ(γ−1) > 0 for merely holding range — i.e. it would pay to stay FAR (a farm worse
#     than the one we are fixing). γ=1 makes holding range exactly 0-reward: only net closure pays.
K_CLOSE = 120.0  # weight on the closing-range potential (telescoping; γ=1 → non-farmable)
R_CLOSE_MIN = 150.0  # curriculum-82: LOWERED 250→150. The chase-BFM autopsy (`_diag_chase_bfm.py`, BASELINE
# vs the CURRENT aggressive T3) showed the policy fires 0 rounds inside 250 m: the closing PBRS (K_CLOSE=120,
# the largest dense weight) CLIPPED at 250 m gave ZERO gradient to press into the gun-kill CONTROL ZONE
# (~150-200 m, where sigma_track opens to 2.75-3.0° so its ~3° lead finally connects). Greedy stalled at 250 m
# firing un-hittable 3-7° shots from a 400-600 m standoff; MPC rammed in and overshot (ω 76-126°/s, 52-74° lead).
# A reward VALLEY (the 250-400 m overshoot transient, ω 23°/s, lead grows) sat between the standoff optimum and
# the control zone with no pull across it. Extending the clip to 150 m fills that valley so the policy can cross
# into the zone where the outcome-mode settle shaping (v22 lead-null / v23 low-closure / convert-dwell) operates.
# Ram is guarded NOT by a high clip but by v23's in-zone low-closure gate: the extended pull is paid ONLY at
# |closure| ≤ CONVERT_CLOSURE_MAX → a CONTROLLED lag press, not the MPC ram. 150 m = the finish band / the old
# pre-v18 CONVERT_RANGE_MIN. The one lever never touched in curriculum-01→81 (R_CLOSE_MIN was 250 throughout).

# --- reward-v8 (2026-08-27): potential-based AIM shaping on the WEZ gun-solution quality.
#     GROUND-TRUTH greedy eval of the reward-v7 stage-0 milestone (which "gated" at 70% in the
#     training log) came back 0/20 kills even at POINT-BLANK 400 m — every episode BYTE-IDENTICAL:
#     the deterministic policy flies ONE firing pass (chips ~26 HP), drifts out of the gun
#     solution, then COASTS to the clock. It never RE-ACQUIRES. So the training "kills" were the
#     guided-fire OVERRIDE forcing the trigger (which does not transfer to the greedy prior), and
#     the bottleneck is not range/aim/commitment but that after the first pass there is NO dense
#     gradient rewarding getting the pipper back on: the one solution-holding term
#     (K_SOLUTION·gun_quality) is CAPPED at TRACK_BUDGET — banked in the first ~6 s. Uncapping it
#     is unsafe: gun_quality is nonzero out to R_EFF (1200 m; ≈0.42 at 700 m nose-on), so a passive
#     tail-chase could bank K_SOLUTION·gun_quality for a whole episode → a farm far larger than a
#     kill (the very trap TRACK_BUDGET was added to kill). This instead adds a POTENTIAL-BASED term
#     F = K_AIM·(q′ − q) on the gun-solution quality q (γ = 1, exactly like the closing term): it
#     PAYS to sharpen the firing solution and COSTS to lose it — supplying the missing re-acquire
#     gradient — yet telescopes to K_AIM·(q_end − q_start) over an episode, so it CANNOT be farmed
#     by holding a solution and does not shift the optimal policy (it only densifies the sparse
#     converge→hold→finish path). γ = 1 (not γ<1) for the same reason as the closing term: a γ<1
#     PBRS on a bounded-positive potential pays a standing bonus γq−q < 0… — i.e. it would COST to
#     merely hold aim (discouraging the sustain we want); γ=1 makes holding aim exactly 0-reward,
#     so only net improvement/loss of the solution pays. Applied inside reward.compute (it has both
#     transition states). Uncapped (a PBRS must telescope freely; capping it breaks that).
K_AIM = 60.0  # weight on the gun-solution-quality potential (telescoping; γ=1 → non-farmable)

# --- reward-v13 (2026-08-28): potential-based STERN-CONVERSION shaping — the "turn the merge into a
#     chase" gradient. GROUND-TRUTH greedy eval of the c35 (reward-v12) stage-3 milestone: a decisive
#     killer at ≤60° (24/40 @45°, 15/40 @60°, ammo to spare) but 0/40 kills @≥75° — it deals 0 HP and
#     mag-dumps empty air. Root cause: a GRADIENT HOLE between a high-aspect merge and the gun WEZ. The
#     closing PBRS (K_CLOSE) pays to reduce RANGE and the aim PBRS (K_AIM) pays to sharpen a solution
#     that is nonzero only INSIDE R_EFF — but NOTHING rewards getting into the rear-quarter GEOMETRY in
#     between, so the policy closes range nose-off (chasing a beam bandit it cannot yet point at), ends
#     up co-alt abeam, and sprays. This adds F = K_ASPECT·(Ψ(A′) − Ψ(A)) on the BANDIT ASPECT A (rg.aspect;
#     LOW = in his rear quarter), with Ψ(A) = −min(A, PURSUIT_GATE_HI)/PURSUIT_GATE_HI ∈ [−1, 0]: flying to
#     his six (A↓) PAYS, falling off it (A↑) COSTS — the missing "convert the merge to a stern chase" pull.
#     Same design as the K_CLOSE/K_AIM PBRS: γ = 1 → telescoping (episode sum = K_ASPECT·(Ψ_end − Ψ_start)),
#     so an aspect wobble nets 0 (non-farmable) and it is policy-invariant; it caps at K_ASPECT (< R_KILL),
#     so a kill stays strictly dominant and parking at his six (no close/fire) is −EV vs the draw penalty.
#     WHY the bandit aspect (not my ata): aspect depends on the BANDIT's heading, so the policy cannot lower
#     it by pointing away — only by actually flying to his six, the exact skill the c35 policy is missing.
# --- curriculum-93 (2026-09-05): 80.0 → 144.0, the SLOPE-PRESERVING partner of the
#     PURSUIT_GATE_HI 100° → 180° un-clip below. Ψ's gradient is K_ASPECT/PURSUIT_GATE_HI per radian, so
#     widening the clip alone would DILUTE the stern-conversion pull everywhere (80/rad(100°) = 45.84 →
#     80/rad(180°) = 25.46, a 0.55× tax on the one PBRS the chase finish depends on). Scaling by the same
#     180/100 keeps it at 144/rad(180°) = 45.84 — bit-identical below 100° aspect, purely ADDITIVE over
#     100–180°. Change these two together or not at all.
#     KNOWN COST, stated because slope-preservation does NOT cover it: the CAP rises with the weight, so a
#     full 180°→0° conversion now pays up to 144 (was 80) and "convert then park" scores 144 + R_TIMEOUT
#     = +94 (was +30). This does NOT invert the incentive — the term telescopes and is paid in BOTH the
#     kill and timeout branches, so the MARGINAL value of finishing is R_KILL − R_TIMEOUT = 200 either
#     way (PBRS is policy-invariant; it reshapes the gradient, not the optimum). What it does raise is the
#     shaping term's size relative to the outcome terms, so watch the verdict's timeout/kill split.
K_ASPECT = 144.0  # weight on the stern-conversion (bandit-aspect) potential (telescoping; γ=1 → non-farmable)

# --- reward-v15 (2026-08-29): WOUNDED-BANDIT re-attack amplifier — the "chase down the enemy after
#     losing it" fix. A per-step FINISH diagnostic of the c41 45° killer (_diag_finish.py) showed the
#     frontier stall is NOT fire-discipline or aim precision: at a 55° start it lands a crushing first
#     pass (bandit → ~18 HP) then OVERSHOOTS, separates to a ~1.25 km / gun_quality≈0 lazy orbit, and
#     COASTS for 55 s firing ZERO of 424 remaining rounds — it loses the wounded bandit and never chases
#     it back down (the exact behaviour the user flagged). Nothing made re-attacking pay more than
#     coasting: R_TIMEOUT only bites at t=60 s and the K_CLOSE/K_AIM/K_ASPECT PBRS telescope (a wide
#     orbit nets ~0). This multiplies those three "get into a kill position" PBRS terms by a wound factor
#     wf = 1 + WOUND_GAIN·clip((WOUND_HP − bandit_hp)/WOUND_HP, 0, 1) whenever the bandit is ALIVE and
#     below WOUND_HP, so closing / sharpening the solution / converting to his six on a nearly-dead bandit
#     PAYS up to (1+WOUND_GAIN)× more — a strong pull to finish. It scales BOTH signs of each telescoping
#     term by the SAME wf (bandit HP is monotone non-increasing within an episode, so wf is ~constant over
#     any close/open cycle at fixed HP) → still non-farmable; it is a POSITIVE shaping term (no standing
#     cost) so there is NO perverse "fly out of the bubble to escape a penalty" incentive. Wound-gated →
#     a STRICT no-op vs a healthy bandit: the merge and the working 20-45° regime are untouched (the
#     45° terminal phase lasts 0.1 s, so wf>1 fires for ≤3 ticks there), and it never fires at 60°
#     (bandit never drops below 62 HP) — 60° is a separate first-pass-conversion problem.
WOUND_HP = 35.0     # bandit health below which the re-attack amplifier engages (finishable / "wounded")
WOUND_GAIN = 1.5    # max extra multiplier on the close/aim/aspect PBRS at bandit_hp → 0 (wf up to 2.5×)

# --- reward-v11 (2026-08-28): ASPECT-GATE on the immediate-pursuit dense terms — the "you can't
#     start pure pursuit immediately at high aspect" fix. reward-v10 conserves ammo and cleanly
#     kills at ≤45° start aspect, but at a 60–90° (beam→head-on) merge the greedy policy still pulls
#     PURE PURSUIT straight at the bandit, overshoots the turn circle, and never converts — because
#     the dense shaping actively STEERS it there: the non-telescoping nose-on tracking term rewards
#     "point at the bandit NOW" every tick, and the closing PBRS, under the RL discount, rewards
#     "close SOONER" — both are maximised by an immediate lead-collision rush, which is exactly the
#     overshoot. The BFM fix is a LEAD TURN first (turn to the inside of his circle, arrive behind),
#     which those terms transiently PUNISH (you briefly point ahead of him and hold/open range).
#     This gates the two immediate-pursuit terms by a "pursuit-appropriateness" factor f_pursuit(A)
#     from the BANDIT ASPECT A (rg.aspect: angle at the bandit's tail to the LOS — LOW = I am in his
#     rear quarter, HIGH = I am abeam/ahead of him at the merge). f_pursuit = 1 for A ≤ PURSUIT_GATE_LO
#     (clearly behind → pursue freely, unchanged), ramping down to PURSUIT_FLOOR for A ≥ PURSUIT_GATE_HI
#     (beam/head-on merge → the nose-on-and-close bribe is largely removed). The policy then earns the
#     FULL pursuit reward only once it has MANEUVERED into the bandit's rear (A drops) — i.e. the
#     reward gradient now points to the lead turn, not the overshoot. WHY the bandit aspect and not my
#     own nose-to-LOS (ata): aspect depends on the BANDIT's heading, so the policy cannot cheaply lower
#     the gate by pointing away — only by actually flying to his six, the skill we want. NON-REGRESSION
#     / SAFETY: (1) strict no-op at low aspect (A ≤ LO → f_pursuit = 1), so the working ≤45° kills —
#     which end up well inside the rear quarter — are untouched. (2) Only the NON-telescoping nose-on
#     tracking term and the CLOSING-TOWARD (positive) part of the closing PBRS are damped; the PBRS's
#     opening (negative) part keeps its full honest cost, so no close-low/open-high cycle nets positive
#     (every closed range cycle still sums ≤ 0 → the term stays non-farmable). (3) The gun-solution
#     terms (K_SOLUTION·gun_quality, the K_AIM aim PBRS) and damage/kill are NOT gated — a real shot
#     pays fully at ANY aspect, so the kill incentive is intact; only the "rush the merge nose-on"
#     bribe is removed. Warm-started from the c33 (reward-v10) best; watch the 60–90° verdict for the
#     lead-turn conversion (kills, not chip-and-timeout) with the ≤45° kills preserved.
PURSUIT_GATE_LO = radians(50.0)  # aspect ≤ this ⇒ full immediate-pursuit reward (clearly in his rear quarter)
# --- curriculum-93 (2026-09-05): 100° → 180° = UN-CLIP the stern-conversion potential. Ψ(A) =
#     −min(A, PURSUIT_GATE_HI)/PURSUIT_GATE_HI is CLIPPED FLAT above PURSUIT_GATE_HI, so over the whole
#     100–180° aspect range dΨ/dA is EXACTLY 0.000/deg — the reward is blind to the difference between
#     "abeam at 100°" and "he is pointing straight back at me at 180°", and the only dense term that pays
#     for converting to a stern chase gives no gradient at all in the region the policy actually fails in.
#     Localised, not guessed: the curriculum-91 measurement of the failure state is aspect median 147°,
#     and curriculum-92 re-measured 154–158° — i.e. the fight lives ENTIRELY inside the flat region. This
#     is the same defect shape as curriculum-82's R_CLOSE_MIN 250→150 (a clipped K_CLOSE flat-spotted the
#     close reward), which produced the first forward motion in ~17 ceilings. 180° is the natural bound:
#     aspect cannot exceed it, so no clip remains anywhere and no new tunable is introduced.
#     CONTAINMENT (verified, and the reason this is one variable): PURSUIT_FLOOR = 1.0, so all three
#     f_pursuit branches in reward.py return exactly 1.0 and neither this constant nor PURSUIT_GATE_LO can
#     reach f_pursuit — this feeds ONLY _psi. ⚠ That containment is conditional on PURSUIT_FLOOR staying 1;
#     re-arming the floor (as curriculum-80 briefly did) makes this a gate width again.
PURSUIT_GATE_HI = radians(180.0)  # aspect ≥ this ⇒ reward floored (the beam/head-on merge — don't bribe the rush)
# --- reward-v12 (2026-08-28): NEUTRALISE the v11 aspect gate (PURSUIT_FLOOR 0.2 → 1.0 ⇒ f_pursuit ≡ 1
#     at every aspect, an exact no-op of v11). WHY: the user's acceptance behaviour is now "survive the
#     merge, DON'T deck yourself, then CHASE the bandit down and empty your gun into it — the KILL is the
#     goal, the chase is the means." The v11 gate damped the nose-on + closing-toward reward to 20% at high
#     aspect precisely to STOP a premature lunge into the beam — but that is the SAME geometry as the
#     RE-ATTACK TURN back onto a bandit you just overshot, so it also suppressed the chase we now want.
#     Two curricula under v11 (c33/c34) confirmed the failure: the 19M policy learned to SURVIVE the
#     75–90° merge (0 decks) but then flew PASSIVE — spent 0 rounds, dealt 0 HP, timed out — because the
#     chase gradient was damped and a passive draw was cheap. The user has deprioritised the pure
#     pure-pursuit-into-the-beam kill (v11's raison d'être; an expert can't do it either), so the gate's
#     cost now outweighs its benefit. Restoring f_pursuit ≡ 1 makes turning back onto a fleeing bandit pay
#     at any aspect. STRICT no-op at low aspect is unchanged (was already 1). Overshoot risk is left to the
#     honest levers that don't suppress the chase: the reward-v10 ammo-waste ramp + the reward-v9 deck
#     cushion. The gate code + PURSUIT_GATE_* stay in place (documented) so a smarter merge-vs-re-attack
#     gate can be reintroduced later without a code change if the raw rush re-emerges. See reward-v12 in
#     reward.py and the engagement-conditioned timeout (PASSIVE_DRAW_HP / PASSIVE_TIMEOUT_MULT) below.
# --- curriculum-80 (2026-09-02, from-scratch): RE-ARM the aspect gate at PURSUIT_FLOOR 1.0 → 0.3. The
#     16-run merge program's dominant failure is deck-OR-DRAW; with f_pursuit ≡ 1 the dense nose-on term
#     (K_TRACK·f_nose·f_range) AND the positive K_CLOSE bribe pay FULL at the ~180°-aspect head-on merge,
#     rewarding the lead-collision rush that MANUFACTURES the crossing velocity (the geometric draw). v12
#     floored it because the SAME high-aspect damping also suppressed the re-attack CHASE — but that
#     conflated two rewards: the nose-on "point at him NOW" bribe (WRONG for a crossing target — you want a
#     lead-turn to his FUTURE six, not to nose-chase the cross) vs the geometric conversion, which is paid
#     by the UNGATED K_ASPECT stern-conversion PBRS. So damping K_TRACK/K_CLOSE at high aspect is correct
#     for BOTH the merge and the re-attack; K_ASPECT keeps the chase aggressive from any aspect. 0.3 (not
#     v11's 0.2) leaves a floor so the chase never goes cold; if the P1 chase verdict shows passivity
#     (spent 0 / dealt 0 — v12's signature) raise to 0.5. Cold from-scratch ⇒ no warm peak to disturb.
# --- curriculum-81 (2026-09-02, WARM-ADAPT): REVERT to v12's 1.0. c80's cold-from-scratch reframe is
#     disproven (P1 honest-σ + P2 loose-σ both fail BASELINE at 5M in greedy AND MPC — BASELINE is the
#     ENDPOINT of the whole warm curriculum-01→63 chain, not a single cold run). The chosen path is now
#     WARM-ADAPT from `runs/BASELINE_merge_survivor_2480k.pt` on the `chase-t3-killgate` ladder vs the
#     current [[aggressive-t3-break]] T3 — keep the ≤55° finish, adapt to the harder opponent, re-earn
#     clean kills. That means MATCHING BASELINE's reward landscape: BASELINE (c63) trained with f_pursuit≡1
#     (v12). 0.3 was a MERGE-rush damper (aspect≈180) irrelevant on this [0,90] CHASE ladder, and at the
#     50-90° frontier rungs it re-taxes exactly the pursuit turn warm-adapt needs MORE aggressive. So 1.0.
PURSUIT_FLOOR = 1.0  # curriculum-81 warm-adapt: f_pursuit ≡ 1 (v12) — matches the BASELINE reward; the chase is undamped at every aspect.

# --- reward-v12 (2026-08-28): ENGAGEMENT-CONDITIONED clock-timeout penalty (the other half of the
#     chase fix, applied in DogfightGymEnv.step where the timeout is seen). The flat per-stage timeout
#     penalty (R_TIMEOUT = −40) gave ZERO gradient between "flew passive to the clock" and "chased hard
#     but couldn't finish" — so the 19M policy picked the safe passive draw at 75–90°. This splits the
#     draw: a PASSIVE standoff draw (the learner dealt < PASSIVE_DRAW_HP over the whole episode — it never
#     ran the bandit down) is penalised PASSIVE_TIMEOUT_MULT× harder, while a hard-fought no-finish draw
#     (chased + CONNECTED for real damage but couldn't gun the kill) keeps the milder base penalty. It
#     keys on DAMAGE DEALT (real hits only) — NOT ammo spent — so it cannot be gamed by mag-dumping from
#     standoff (the reward-v10 ramp keeps that spray −EV) and it directly encodes "connect, don't just
#     survive." Terminal + non-farmable. Tension with reward-v7's suicide guard (a timeout worse than a
#     death −60 could make a suicidal merge less-bad than a draw): the harsh −80 bites ONLY the passive
#     draw, and the 19M base already merges at 45–60° with ZERO deaths, so its death-avoidance keeps it
#     from trading passivity for suicide — the realistic escape from −80 is to CHASE + connect (→ base
#     −40) or KILL (+150), not to die. First-cut magnitudes; validate out-of-band and back MULT toward
#     1.5 if decking/suicide re-appears at the high-aspect rungs.
# --- reward-v13 (2026-08-28): FLATTEN the draw penalty. reward-v12's PASSIVE/ENGAGED split created a
#     "spray to clear PASSIVE_DRAW_HP" dodge — the c35 stage-3 policy sprayed its belt at mid-aspect to
#     deal ≥25 HP and settle for the milder −40 engaged draw instead of committing to the kill (0/30
#     kills @65–75° while dealing 16–40 HP). The engaged/passive DISTINCTION is redundant with the dense
#     damage term (DMG_PER_HP=0.5 already credits an 80-HP chase +40 vs a 0-HP standoff), so the terminal
#     split only bought the dodge. MULT→1.0 makes every no-kill clock draw a uniform R_TIMEOUT=−50 (a
#     draw is a failure regardless of chip damage — user intent: "a draw is a failure, the kill is the
#     goal"). −50 stays milder than R_DEATH −60 (death still worst non-deck → no suicide incentive).
PASSIVE_DRAW_HP = 25.0  # HP dealt below which a no-kill clock draw is "passive" (reward-v13: MULT=1 → unused)
PASSIVE_TIMEOUT_MULT = 1.0  # reward-v13: was 2.0. 1.0 → passive == engaged draw (flat R_TIMEOUT); kills the spray-to-25HP dodge

# --- fire discipline (reward-v3): reward-v2's magnitudes (WEZ +0.005 / waste -0.02) taught
#     the agent to NEVER fire — telemetry showed it closing to 856 m and holding a gun
#     solution for ~40% of the episode yet never pulling the trigger, because the in-solution
#     nudge was negligible next to the ~0.016/tick of risk-free tracking it already banked,
#     while the out-of-solution penalty made any exploratory trigger-pull scary. v3 makes
#     firing-in-solution clearly pay (so the policy's trigger mean crosses the fire threshold
#     and the DMG_PER_HP / R_KILL rewards can then take over) and softens the waste penalty so
#     exploratory shots aren't suppressed (ammo is finite — 511 rds — so waste self-limits).
#     Charged per round actually fired (ammo decrement), so it needs no action plumbing.
R_FIRE_WASTE = -0.02  # reward-v13: was −0.005. c35 pulsed the trigger to dodge the burst-ramp and still mag-dumped empty air at aspect ≥70° (511 rds, 0 HP); a 4× base cost makes even PULSED out-of-WEZ spray clearly −EV while the +0.05 WEZ bonus (2.5×) keeps a real in-solution shot attractive (no reward-v3 never-fire regression — the warm policy already fires in-WEZ at low aspect)
R_FIRE_WEZ = 0.05  # per round fired INSIDE the solution — a real incentive to shoot when in the WEZ

# --- reward-v9 (2026-08-28): two curriculum-29 fixes for the c27 champion's self-inflicted losses.
#
# (1) DENSE HARD-DECK CUSHION. R_DECK (−200) is TERMINAL — a cliff with no gradient warning the
#     policy off a nose-low chase until it is already dead. The c27 champion's losses at the wide
#     (near-beam) aspects were dominated by self-inflicted DECKINGS during nose-low reversals /
#     extends (within-episode time-series: hard_deck at ~1100 m mid-fight, not a gun death). A
#     terminal-only penalty gives credit assignment nothing to chew on until the bust tick. This
#     adds a dense per-tick cost that ramps 0 → K_DECK_FLOOR as AGL falls from (DECK_AGL +
#     DECK_MARGIN) down to DECK_AGL, so descending into the cushion is felt continuously and the
#     policy trades a marginally tighter track for staying alive. It only bites in the bottom
#     ~800 m (starts at 15 000 ft), so it does NOT tax normal-altitude fighting, and it is kept
#     milder per-tick than the tracking gradient (K_TRACK+K_SOLUTION ≈ 0.04) so it SHADES a
#     committed low gun pass, never forbids it. Applied in reward.compute (needs only current AGL).
K_DECK_FLOOR = -0.02  # max per-tick penalty at the deck edge (ramps in over DECK_MARGIN above DECK_AGL)
DECK_MARGIN = 500.0  # m; cushion above the hard deck over which the floor penalty ramps 0 → K_DECK_FLOOR
#
# (2) FIRE-GATE = the EXPERT's HIT gate. The fire bonus/waste split used the bare WEZ LEAD CONE
#     (in_range AND lead_error < FIRE_LEAD_RAD = 4°), which pays the +R_FIRE_WEZ bonus for firing
#     at LONG range inside a 4° cone — where the 8 mrad dispersion footprint (and any lead error)
#     dwarfs the target so the whole magazine misses. That rewarded dumping all 511 rounds on the
#     first pass. reward-v9 redefines "in solution" to MIRROR the PursuitExpert's proven hit gate:
#     rng·sin(lead_error) < GUN_FIRE_MISS_M (10 m) — the predicted lateral miss falls inside the
#     target's ~8 m hit sphere, i.e. exactly where rounds connect and where the expert (which kills
#     the T3) actually fires. Long-range sprays now flip from +R_FIRE_WEZ to R_FIRE_WASTE, so ammo
#     discipline emerges without RAISING the waste penalty (kept at −0.005; the reward-v3 note warns
#     −0.02 taught the policy to NEVER fire — the gate, not a harsher penalty, is the honest lever).

# --- reward-v10 (2026-08-28): BURST-RAMP waste penalty (fire discipline / ammo conservation).
#     reward-v9's HIT gate was necessary but NOT sufficient: a greedy-eval trace of the c29 champion
#     at aspect 60° shows it empties its ENTIRE 511-round magazine in the first ~7 s — ~41 rounds
#     sprayed pre-merge at gun_quality 0, a genuine ~48-round IN-solution burst during the t=4.4–4.6 s
#     kill window (dealing 48 HP), then ~233 more dumped at gun_quality 0 after the overshoot — and
#     flies the remaining ~47 s DISARMED, so it can re-close (down to 338 m by t=25 s) but can never
#     finish. reward-v9 correctly flips those sprays to R_FIRE_WASTE, but a flat −0.005 makes a
#     460-round spray cost only −2.3 — toothless next to R_KILL (150). The two prior single-scalar
#     attempts on this axis BOTH failed: v2's flat −0.02 taught NEVER-fire; v9's tighter gate at flat
#     −0.005 still sprays (the champion IS reward-v9). So the lever is not the flat magnitude but the
#     SHAPE: the failure mode is HOLDING the trigger through a long out-of-solution burst, not firing
#     per se. This charges a per-round waste cost that ESCALATES with the length of the CURRENT
#     out-of-solution burst — cost/round = R_FIRE_WASTE · min(1 + streak/WASTE_RAMP_TAU, WASTE_RAMP_MAX)
#     — where `streak` is the running count of consecutive wasted rounds. It RESETS to 0 on any
#     in-solution round (a hit-gate round is disciplined + rewarded, not penalised) and DECAYS while
#     not firing (WASTE_STREAK_DECAY/tick — a genuine trigger release / reposition forgives). Key
#     property: at streak 0 the cost is EXACTLY R_FIRE_WASTE (unchanged from v9), so a short
#     exploratory burst is as cheap as today — this is the structural guard against the v2 never-fire
#     trap, not a hope. Only a SUSTAINED dump ramps: a ~233-round held-trigger spray costs ≈ −15
#     (≈ 10 % of a kill) instead of −1.2, making the mag-dump clearly −EV while a disciplined
#     in-solution burst still nets the damage + kill. reward.compute stays a pure function — it takes
#     the incoming `waste_streak` and returns the updated one in info; DogfightGymEnv.step owns the
#     per-episode state (exactly like TRACK_BUDGET's `_track_used`). Warm-started from the c29
#     champion (which already fires aggressively in-solution), so the risk is over-pruning the spray,
#     not re-triggering never-fire. Magnitudes are a first cut — validate out-of-band (ammo remaining
#     at kill, beam net-ahead) and steepen WASTE_RAMP_TAU / raise WASTE_RAMP_MAX if the spray persists.
WASTE_RAMP_TAU = 6.0      # reward-v14: was 10.0. Secondary to the GUN_FIRE_MISS_M tightening — once the grazing spray flips to waste, reach a −EV multiplier faster (a burst hits 2× after 6 wasted rounds, ~34× by 200) so a sustained mag-dump is punished swiftly
WASTE_RAMP_MAX = 40.0     # cap on the waste multiplier (max per-round waste = R_FIRE_WASTE · this)
WASTE_STREAK_DECAY = 2.0  # reward-v13: was 6.0. Slower decay so a brief trigger release no longer forgives the burst — closes the "pulse the trigger to keep the ramp low while still mag-dumping" loophole c35 exploited at high aspect

# --- reward-v16 (2026-08-30): ANTI-SPIRAL attitude penalty — the user's directive after the 7th
#     merge-family collapse (c49-c57): "don't teach it to spiral DOWN; a real pilot does a vertical egg."
#     Every c49-c57 deck shares one signature: after an overshoot the policy hauls the nose toward the
#     bandit while banked, and at high bank a hard pull's lift vector is mostly SIDEWAYS — so the jet
#     DESCENDS through the turn (a nose-low corkscrew) and busts the deck. The imitation override tried to
#     cure this by CLONING a scripted turn, but the greedy policy never internalised it (c56/c57 decked
#     11-12/12 while the train log showed deck=0). A REWARD gradient needs no cloning: it reshapes the
#     policy's OWN behaviour. This taxes exactly the descending-spiral attitude — NOSE-LOW × BANKED —
#     per tick, so the policy is pushed to reverse in the VERTICAL (pull the nose up, egg over the top)
#     instead. Design guards: (1) it is GATED OFF in a real gun solution (gun_quality ≥ SPIRAL_SOLUTION_Q)
#     so a committed diving guns pass on a lower bandit is never taxed; (2) it is a nose-low×bank PRODUCT,
#     so a wings-level dive (bank→0) OR a nose-HIGH banked turn (the egg / lead turn; nose-low→0) pays
#     NOTHING — only the corkscrew is taxed; (3) it is NOT a climb bonus — rewarding altitude would breed
#     energy-hoarding passivity (the opposite of the anti-cowardice intent), so this only makes the spiral
#     COST accrue; (4) mild per-tick (= the deck-floor scale) so it SHADES, never forbids, a hard break.
#     CONGRUENT with the champion's nose-high energy-fighter identity → reinforces (not fights) the
#     co-heading finish it must not forget (there it tracks nose-on [gated off] or zooms nose-high [nose-
#     low→0], so the penalty rarely bites) — the c58 bet that this finally lets the merge + finish coexist
#     in one network. Stateless (needs only me.theta/me.phi + rg.gun_quality) → lives in reward.compute.
#
# --- reward-v16b (2026-08-30): ALTITUDE-GATE the anti-spiral penalty. c58 verdict #3 @5.0M found global
#     PASSIVITY (cowardice): 55° co-heading finish collapsed 9/12 kill → 0/12 with 0 damage, stage-1
#     stalled 2.3M steps at net 0% (survives the merge, fires 0 rounds, all timeouts). Root cause: the v16
#     penalty fired at EVERY altitude, so it taxed the legitimate nose-low BANKED converting pursuit turn
#     (the aggressive re-attack the user explicitly wants) just as hard as a genuine low-altitude death
#     spiral — the policy learned to avoid the taxed attitude by not converting at all. But a death spiral
#     is a LOW-ALTITUDE event: nose-low + banked at fighting altitude is textbook BFM (a descending lead
#     turn / low yo-yo), only dangerous when there is no room to recover. Fix: ramp the penalty in only as
#     AGL falls from SPIRAL_ALT_BAND_AGL down to the deck cushion (DECK_AGL + DECK_MARGIN), where the
#     deck-floor term then takes over. Above the band the corkscrew attitude pays NOTHING → aggressive
#     turns at altitude are freed (cures the passivity); down low it still bites (keeps the anti-deck
#     safety net that broke the 7-collapse pattern). Anti-cowardice directive preserved: KILL stays the
#     goal, the converting turn is the MEANS, and now only a genuinely deadly low corkscrew is penalised.
K_SPIRAL = -0.02             # max per-tick penalty for a fully nose-low + fully banked (descending-spiral) attitude
SPIRAL_PITCH_RAD = radians(45.0)   # nose-DOWN pitch at which the nose-low factor saturates to 1 (linear from 0 at level)
SPIRAL_BANK_MIN_RAD = radians(30.0)  # |bank| below this ⇒ not "spiraling" (bank factor 0 — a wings-level dive pays nothing)
SPIRAL_BANK_MAX_RAD = radians(75.0)  # |bank| at/above this ⇒ bank factor saturates to 1
SPIRAL_SOLUTION_Q = 0.3      # gun_quality ≥ this ⇒ committed guns pass → penalty GATED OFF (never tax a real diving shot)
SPIRAL_ALT_BAND_AGL = 2000.0  # reward-v16b: AGL ceiling above which the penalty is 0 (free aggressive turns at fighting
                             # altitude); ramps 0→1 from here down to the deck cushion (DECK_AGL+DECK_MARGIN=800m)

# --- reward-v17 (2026-08-30): the SURVIVE-CONVERT terminal — the two-policy MERGE architecture.
#     After ~9 curricula (c49-c59) a single SHARED network provably cannot hold BOTH the head-on
#     merge AND the ≤55° chase-finish: every warm-champion merge attempt either DECKS or goes
#     PASSIVE and always ERODES the champion's finish (destructive interference — a confirmed method
#     ceiling, see the curriculum-59 memory). The fix is architectural, not another reward tweak:
#     FREEZE the champion as the FINISHER (F, already beats T3 at every ≤55° chase geometry, 0
#     losses) and train a SEPARATE merge-CONVERTER policy (M) whose ONLY job is to survive the pass
#     and CONVERT to a rear-quarter firing position behind a LIVE bandit — at which point a geometric
#     selector HANDS THE JET OFF to the frozen F to finish. M carries no finish burden, so there is
#     no interference to erode; F never trains, so it can never degrade. A good pilot converts the
#     merge routinely — the scripted experts only decked because untrained for it (see the
#     merge-kill-not-scripted memory), so an RL M with a clean survive+convert objective should learn
#     the reversal the shared net never could.
#
#     The terminal fires (in reward.compute) ONLY when `convert_bonus > 0` (M's runs; a strict no-op
#     for every legacy ladder, which passes convert_bonus=0). "Converted" = a genuine pursuit
#     position: I am in the bandit's REAR QUARTER (rg.aspect ≤ CONVERT_ASPECT_RAD — his tail toward
#     me), my NOSE is ON him (rg.ata ≤ CONVERT_ATA_RAD — NOT merely swept geometrically behind while
#     flying away; this ata gate is what stops a head-on FLY-THROUGH from spuriously "converting"),
#     the RANGE is in F's prosecution band (CONVERT_RANGE_MIN ≤ rg.rng ≤ CONVERT_RANGE_MAX), and the
#     bandit is ALIVE. The episode TERMINATES there (+convert_bonus), so M never sees the finish (F
#     does) and M's value function learns "reach the rear quarter = win." M is also NOT paid the fire
#     WEZ bonus (reward.py gates it off when convert_bonus>0) — M holds fire, saving the shared jet's
#     ammo for F. A convert_bonus ≈ R_KILL (150) makes converting +EV even under a substantial merge
#     death-risk (150·0.7 − 60·0.3 = +87 ≫ the −50 timeout of a coast) so passivity is not the
#     risk-optimal play the way it is for the finish-burdened shared net.
# --- reward-v18 (2026-08-30): the SETTLED, FINISHABLE hand-off (fixes the two-policy seam).
#     The c60 verdict exposed the reward-v17 box as too permissive: greedy M "converted" 100% at the
#     135° merge, yet the frozen finisher F then DECKED 7/12 and killed 0/12. The hand-off telemetry
#     showed why — EVERY "conversion" fired at ~3 s (DURING the initial merge pass, not after a real
#     post-merge turn) with the jet BANKED ~90-104° (knife-edge/slightly inverted) and the geometry
#     clipping the FAR CORNER of the box (aspect ≈50°, ata ≈57°). That is a merge-pass TRANSIENT, not a
#     settled arrival. F, handed an unflyable knife-edged jet (and having NO anti-spiral training — that
#     was M's tool), spiralled from 4530 m into the deck. The champion finishes fine from its CLEAN,
#     wings-level ~45-55° spawns (its 8/12 record) — it is the 100° bank + corner-clip that breaks the
#     finish, not the aspect or the altitude. So the box must demand a hand-off inside F's proven
#     envelope: (1) NOSE genuinely on him (CONVERT_ATA_RAD 60°→40°, rejecting the ata≈57° corner-clip),
#     (2) the jet SETTLED / roughly wings-level so F inherits a flyable state and does not spiral
#     (NEW CONVERT_BANK_RAD = 60°, rejecting the 90-104° knife-edge slices), (3) range nudged up off the
#     merge singularity (CONVERT_RANGE_MIN 150→250). ASPECT stays 50° — F demonstrably finishes clean
#     from there; the killer was the attitude, not the aspect. This trains M to ROLL OUT behind the
#     bandit before the hand-off fires (the real conversion skill) instead of clipping the corner mid-slice.
CONVERT_ASPECT_RAD = radians(50.0)  # bandit aspect ≤ this ⇒ I'm in his rear quarter (= PURSUIT_GATE_LO anchor)
CONVERT_ATA_RAD = radians(40.0)     # reward-v18: nose genuinely ON him (was 60° — rejected the ata≈57° corner-clip)
CONVERT_BANK_RAD = radians(60.0)    # reward-v18: |bank| ≤ this ⇒ jet SETTLED/flyable → F inherits a state it can finish (not a 100° knife-edge)
CONVERT_RANGE_MIN = 250.0           # reward-v18: nudged up off the merge singularity (was 150)
CONVERT_RANGE_MAX = 1000.0          # m; above this F inherits too loose a chase (champion trained ~700 m)

# --- reward-v19 (2026-08-31): the OUTCOME reward for the BFM head-on program (curriculum-62). ---
#     The BFM pivot (see bfm-library-coach memory / the user directive "THESE BFM are how we win the
#     fight, not this pure pursuit thing") abandons the dense pure-pursuit shaping that walled ~18
#     reward versions and every merge attempt (c33-c61: shared-net AND two-policy method ceilings).
#     When `outcome_mode` is set (env flag, run-level like convert_bonus), reward.compute ZEROES the
#     five dense pursuit terms — K_TRACK, K_SOLUTION (the nose-on tracking pay), K_CLOSE (closing
#     PBRS), K_AIM (gun-solution PBRS), K_ASPECT (stern-conversion PBRS) — the whole "point at him /
#     close now" gradient that bribes a flat energy-dumping turn tighter than the sustainable radius
#     ("a straight merge is always a death"). What remains is OUTCOME-shaped only: the sparse damage
#     term (real HP off him), the terminals (kill/death/deck/trade/bubble/timeout), fire discipline
#     (WEZ bonus + waste ramp — rewards HITS, not aim), the anti-spiral + deck-cushion safety nets,
#     PLUS two new OUTCOME terms below. The BFM-coach exploration override (train_parallel
#     --bfm-override-p0) supplies the deck-free, BFM-literate trajectory distribution the sparse
#     reward needs to learn from — RL then discovers the CONVERSION the hand-coded coach could not.
# (1) SUSTAINED rear-quarter conversion DWELL: a small per-sim-tick reward for HOLDING the SETTLED
#     rear-quarter firing box (his six + nose roughly on + wings settled + in the gun band) — reusing
#     the reward-v18 CONVERT_* box. This is the "sustained-tail-conversion" outcome the user named:
#     it pays ONLY for the converted rear-quarter geometry (aspect ≤ 50° = his tail toward me), NOT
#     for pointing at him at any aspect (the pure-pursuit trap). Capped per-episode at
#     CONVERT_DWELL_BUDGET (< R_KILL) so a kill (which ENDS the dwell) always strictly dominates a
#     park-on-his-six farm — mirrors TRACK_BUDGET's role for the retired tracking term.
K_CONVERT_DWELL = 0.10          # curriculum-83: BOOSTED 0.06→0.10 (close-in settle lever) — per-sim-tick reward
#                                 while in the settled rear-quarter box (both alive). c82 takes fleeting passing
#                                 shots (ω 27°/s) instead of a SUSTAINED track; a richer dwell rewards HOLDING the
#                                 settled WEZ long enough to land the ~50 concentrated hits a kill needs.
CONVERT_DWELL_BUDGET = 80.0     # curriculum-83: RAISED 45→80 per-episode cap. STILL < R_KILL=150 so a kill (which
#                                 ends the dwell) strictly dominates a park-on-six farm (80 vs 150 ≈ 2:1); the
#                                 kill-gated ladder further blocks any dwell-farm-without-kills from advancing.
# reward-v20 (2026-08-31, curriculum-65 "terminal settle-and-shoot"): the outcome-mode dwell uses its
#     OWN range floor, decoupled from the (dead) two-policy CONVERT_RANGE_MIN=250. The nose-to-nose
#     merge trace of the deliverable (_peak_2480k) showed greedy WINS the angles (best_asp 1-13°,
#     best_ata 0-2°, closest 66-134 m) but never earns settle-reward there because 66-134 m < 250 m,
#     so it swings THROUGH the solution and fires 0 rounds. Dropping the dwell floor to 40 m lets the
#     policy earn "settle" reward for HOLDING that close, wings-settled (|phi|≤60°), nose-on rear-
#     quarter state — exactly where the gun HITS (rng ≪ R_KILL) — so RL is paid to dwell-and-finish
#     instead of merge-pass. Finish-safe: the co-heading finish operates at ~700 m (well inside
#     [40,1000]); this only ADDS reward for a genuinely good close state (bank/ata/aspect still gated).
DWELL_RANGE_MIN = 40.0          # m; outcome-mode conversion-dwell floor (was CONVERT_RANGE_MIN=250)
# (2) SPECIFIC-ENERGY PBRS: F = K_ENERGY·(Φ_E(Es′) − Φ_E(Es)), Φ_E(Es) = clip(Es, 0, 1.5·ENERGY_REF)/
#     ENERGY_REF, Es = AGL + V²/2g (specific energy, m). Telescoping (γ=1) → policy-INVARIANT (Ng et
#     al.): it densifies credit assignment (dumping energy into a flat turn / the deck feels bad NOW,
#     not only at the terminal deck) WITHOUT creating an energy-hoarding basin (a standing altitude
#     reward would — see the reward-v16 note). Encodes "keep your energy → use the VERTICAL" — the
#     core BFM lesson — as a safe shaping term, not a bribe.
K_ENERGY = 25.0                 # specific-energy PBRS gain (telescoping, policy-invariant)
ENERGY_REF = 10000.0            # m; Es normalizer (~6 km AGL + ~280 m/s ≈ 6000 + 4000 = 1.0·ref)
# --- E1 (curriculum-78, safe-core): CORNER-SPEED-BAND airspeed PBRS. A ONE-SIDED telescoping
#     (γ=1 → policy-invariant, non-farmable) potential Φ_band(V) = exp(−½(min(0, V−V_BAND_CTR)/
#     V_BAND_SIGMA)²): FLAT (=1) at/above corner speed and falling off only BELOW it. It pays to
#     REGAIN corner speed and charges bleeding below it, so a yo-yo dip (175→120→175) nets ~0 and
#     regaining energy after a bleed pays — WITHOUT ever taxing carrying or zooming for energy
#     (the flat top has zero fast-side gradient). This is the deliberate fix for the reward-v26
#     overspeed-tax failure ([[curriculum-76-77-bleed-corner]]: taxing speed ABOVE a ceiling made
#     the learner timid/disengage) and it has NO "coast to exactly corner speed" terminal bias
#     (ending fast gives the same Φ=1). Distinct from the reward-v21 CORNER-SPEED FLOOR (a one-tick
#     penalty below V_CORNER_FLOOR): E1 is a smooth telescoping potential that also REWARDS recovery.
#     Reads own airspeed only → no compute() signature change. outcome_mode-gated (legacy no-op).
K_BAND = 20.0                   # corner-band airspeed PBRS gain (telescoping, γ=1; one-sided below corner)
V_BAND_CTR = 175.0              # m/s; = V_CORNER_MPS — the band centre / flat-top knee
V_BAND_SIGMA = 35.0             # m/s; Gaussian falloff width BELOW the knee
# --- reward-v21 (2026-08-31, curriculum-66 "energy-disciplined vertical BFM"): OUT-FLY the T3. ---
#     The T3 opponent trace through the neutral head-on merge (controllers/rl/_t3_trace_tmp.py) proved
#     the merge is a VERTICAL rolling scissors that our jet LOSES on ENERGY: the T3 holds ≥128 m/s in
#     the vertical while our jet over-zooms to a near-stall (airspeed 222→80 m/s) and loses the energy
#     to sustain the converting turn — so it never nulls the crossing (crossing-velocity ceiling) and
#     the merge stays a draw. The absolute-Es PBRS (K_ENERGY) is NEUTRAL on a speed↔alt trade, so the
#     zoom-to-stall costs it NOTHING — the exact hole. Two additions below (both gated to outcome_mode
#     / the BFM program; a strict no-op for every legacy run), plus a hand-off crossing gate:
# (a) CORNER-SPEED FLOOR: a per-tick penalty that bites ONLY below V_CORNER_FLOOR (≈ the T3's sustained
#     ≥128 and just under corner V_CORNER_MPS=175), ramping 0→K_CORNER as airspeed falls from the floor
#     to 0. Taxes bleeding below sustainable maneuvering speed directly (the zoom-to-stall) WITHOUT
#     rewarding raw speed (no run-away bribe) and WITHOUT a standing basin (0 above the floor).
V_CORNER_FLOOR = 150.0          # m/s; below this the jet has bled below sustainable maneuvering speed
K_CORNER = 0.03                 # per-tick corner-speed-floor penalty gain (ramps to this as airspeed→0)
# (b) RELATIVE-Es PBRS: telescoping (γ=1 → policy-INVARIANT, non-farmable; episode sum =
#     K_REL_ENERGY·(Φ_rel_end − Φ_rel_start)) reward for ENDING with MORE specific energy than the
#     bandit; Φ_rel = clip((Es_me − Es_bandit)/ENERGY_REF, −1, +1). The absolute-Es PBRS is blind to the
#     opponent (a symmetric zoom-stall nets ~0), so this shapes RL to WIN the energy war (out-manage the
#     T3 — climb when he's slow, keep speed in the scissors) rather than merely hoard its own energy.
# reward-v24 (2026-09-01, curriculum-75 "long-range neutral-merge ENERGY FIGHT"): the user authorized
#     a PROBLEM REFRAME out of the one-pass merge-kill ceiling — win the way real pilots do: from a
#     fully-neutral long-range co-alt/co-speed start (no handicap), SURVIVE the merge, OUT-ENERGY the
#     aggressive T3 over MULTIPLE turns, and finish from the rear quarter on a LATER pass. Physics: the
#     buffed T3 ([[aggressive-t3-break]]) bleeds energy hard in its 85°-bank AoA-limit break, so an agent
#     that refuses to match it ends faster/higher; against a now-SLOW T3, matching velocity to null the
#     crossing becomes possible — the energy advantage DISSOLVES the crossing-velocity ceiling that made
#     the co-speed merge kill geometric. rel-Es (out-energy the bandit) is the crux of that fight, but at
#     20 it was out-shouted by the dense angle terms (K_AIM 60 / K_ASPECT 80 / K_CLOSE 120, every step),
#     which pull the jet into energy-bleeding angle-fighting (match his break → symmetric bleed → draw).
#     Raised to 50 so "end the fight with MORE energy than him" is a first-class objective competitive
#     with the angle pull — RL is paid to take the lag / keep the vertical / wait for him to go slow,
#     THEN convert. Still telescoping (γ=1 → policy-invariant, non-farmable). Paired with the long-range
#     head-on neutral `energy-merge` ladder + a 180 s multi-turn episode (room for the post-merge fight).
K_REL_ENERGY = 50.0             # relative specific-energy PBRS gain (reward-v24: raised 20→50; telescoping, policy-invariant)
# (c) HAND-OFF CROSSING GATE: the two-policy convert TERMINAL (reward.compute) and the eval selector
#     (_verdict2p_tmp.py) BOTH require LOW |closure| at hand-off, so the frozen finisher F inherits a
#     MATCHED-VELOCITY settled tail (its trained regime) — NOT a high-closure merge-pass fly-through.
#     This is the previously-MISSING "low crossing" conjunct that (per the crossing-velocity finding)
#     was the real, misdiagnosed cause of the c60/c61 seam failure: with only angle+bank gated, the box
#     fired on a ~3 s merge-pass transient (~100° bank clipping the cone corner) at huge closure, and F,
#     handed that knife-edge, spiralled into the deck (7/12, 0 kills). A settled tail chase matched near
#     corner speed has |range-rate| well under this; the ~2×-speed head-on merge pass is 300-500 m/s.
CONVERT_CLOSURE_MAX = 90.0      # curriculum-84: RELAXED 70→90 (from the c83 autopsy). The c83 close-in settle
# lever kept the convert-dwell boost (the WINNER — tracking 43%→85% hittable @250-400m → 13 chase kills @aspect30,
# 54% rate) but TIGHTENED this gate 100→70, which BACKFIRED: the killing-band closure did NOT drop toward the
# hoped ~64 m/s — it ROSE to ~85-87 m/s (250-400m 86.5, 400-600m 84.6). So the 70 gate ended up DENYING reward-v23's
# in-zone bribe for the policy's own natural ~85 m/s converting press, and aspect-45 REGRESSED (c82 killed 2@45 →
# c83 0@45, only 30 HP). 90 sits just ABOVE the observed ~85-87 firing closure so it re-pays that press (recovering
# aspect45) while staying slightly tighter than c82's 100. This gate (1) suppresses reward-v23's in-zone close bribe
# above it and (2) gates the v18 settled-conversion terminal. Warm-continues from c83's 13-kill ckpt. The DWELL boost
# (K_CONVERT_DWELL 0.10 / budget 80) is UNCHANGED — it did the work; this only unblocks the aspect-45 finish.

# --- reward-v26 (curriculum-77 "bleed-to-corner conversion"): REVERTED / DISPROVEN --------------------
#     Hypothesis: the c76 (reward-v25) learner reaches the rear quarter but arrives HOT (249-263 m/s), where
#     full-aft-stick turn rate can't null the ~80° gun lead, so it overshoots and opens out — so tax carrying
#     airspeed above a ceiling while in the firing band to force a high-yoyo / bleed-to-corner. RESULT: the
#     overspeed tax was COUNTERPRODUCTIVE. It made the learner TIMID in the tracking band (won't drive in
#     hard), so it never pressured the T3 into bleeding (c77 bandMinV stayed ~171 m/s vs c76's 92) and it
#     disengaged (3/8 left_bubble) rather than pressing; c77@5.38M had NO low-closure near-solution and was
#     strictly worse than c76's reward-v25 peak (_peak_c76_4730k.pt). Term REVERTED in reward.py; constants
#     kept at 0 gain so nothing inherits the tax. Lesson: don't tax in-zone speed (kills the aggression c76's
#     press-and-slow behaviour needs); the real wall is the 8 m gun hit-radius vs the crossing lead (a lead of
#     3° at 494 m is still a 26 m miss > 8 m) — a geometric/gun-model precision wall, not a shaping gradient.
V_CONVERT_CEIL = 180.0          # m/s; retained for the disproven-term test comment only
K_OVERSPEED = 0.0               # DISPROVEN (was 2.0): overspeed tax removed the aggression c76 depended on

# --- reward-v22 (2026-09-01, curriculum-72 "control-position tracking / null the crossing"): the ---
#     LEAD-ERROR PBRS that fills the gradient hole between a rear-quarter CROSSING and a gun solution.
#     DIAGNOSIS (merge trace + the BFM-coach hot-pass probe): both the trained policy AND the PERFECT
#     scripted BFM coach reach the six o'clock (aspect 4°, 100% conversion) but dump the whole magazine
#     for 0 damage — because at a co-speed crossing the required TOF lead is 30-57°, so gun_quality =
#     range_factor·exp(−½(lead_error/LEAD_SIGMA=3°)²) ≈ 0 AND its GRADIENT ≈ 0 (the Gaussian is flat in
#     its tail). K_AIM (the gun-solution PBRS) therefore gives NO pull out of the crossing — it only
#     bites once lead_error is already within ~10°. Meanwhile K_CLOSE=120 (with PURSUIT_FLOOR=1 → the
#     pursuit gate neutralised) bribes the OVERTAKE at full strength even when behind him, and the
#     overtake IS the crossing. Net: the policy is paid to get aspect→0 + close (the crossing snapshot)
#     and NOTHING is paid to null the crossing (match his velocity so lead_error → 0 = the tracking
#     shot). This is how real BFM wins the gun kill: settle into the control position (in-plane, matched
#     turn rate, LOS-rate → 0) for a TRACKING shot — not a high-crossing flyby snapshot.
#     THE TERM: F = K_LEAD·(Φ_L(lead′) − Φ_L(lead)), Φ_L(θ) = (1 + cos θ)/2 ∈ [0,1] (1 at lead=0, 0.5 at
#     90°, 0 at 180°). Unlike the peaked gun_q Gaussian, Φ_L has a SMOOTH non-vanishing gradient across
#     the WHOLE angle range (dΦ_L/dθ = −sin θ/2, maximal at 90°), so it pulls the policy from a 57°
#     crossing all the way down to a solution. Following that gradient REQUIRES matching the bandit's
#     velocity (that is HOW lead_error drops), so it teaches the settle/velocity-match without cloning a
#     maneuver. GATED to the rear-quarter gun zone (aspect ≤ CONVERT_ASPECT_RAD ∧ rng ≤ CONVERT_RANGE_MAX)
#     so it shapes the CONTROL POSITION, not the approach/merge (a high-aspect crossing there is
#     unavoidable and must not be taxed). ONE-SIDED like K_CLOSE (pay for improvement only in-zone; always
#     charge for degradation) → non-farmable (every lead wobble in/out of the zone sums ≤ 0). K_LEAD=80
#     (= K_ASPECT, < K_CLOSE=120): strong enough to tax uncontrolled overtake yet still lets the jet close
#     to gun range. outcome_mode-gated (a strict no-op for every legacy/non-outcome run).
K_LEAD = 80.0                   # lead-error PBRS gain (telescoping, γ=1 → policy-invariant; null the crossing)
