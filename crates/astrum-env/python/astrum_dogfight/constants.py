"""Reference scales, boundaries and reward magnitudes (design doc §8).

Every value here is a tunable hyperparameter of the *wrapper*, not the sim. They
are collected in one place so a training config can override them wholesale.
Gun ballistics are deliberately absent: lead, range and hit prediction come from
the sim's own fire control (`astrum_env.Env.gun_solution`).
"""

from math import radians

import astrum_env

# --- observation / action shape ---
OBS_DIM = 31
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
AMMO_MAX = float(astrum_env.GUN_AMMO)  # the sim's magazine size

# --- WEZ feature: tracking-error scale on the sim's gun-solution aim error ---
LEAD_SIGMA = radians(3.0)  # rad

# --- episode boundaries the wrapper enforces (§8.4) ---
# The env has no terrain model, so the floor and ceiling are heights above the
# WGS-84 ellipsoid (`alt_hae_m`), not above ground. Over SF Bay the geoid is
# ~−32 m, so the floor sits ~330 m above mean sea level.
DECK_HAE_M = 300.0  # m HAE; hard-deck floor
R_BUBBLE = 15000.0  # m; horizontal engagement-bubble radius from the grid origin
CEILING_HAE_M = 15000.0  # m HAE; engagement-bubble ceiling

# --- Tier-2 scripted opponent: defensive maneuvers (§8.5) ---
R_DETECT = 4000.0  # m; range within which a rear-hemisphere attacker triggers a break
REAR_HEMI_RAD = radians(90.0)  # ATA(me→attacker) beyond this ⇒ attacker is aft of my 3-9 line
V_CORNER_MPS = 175.0  # m/s (~340 kts); corner speed — cap sustained pull to hold it (§8.5)
ALPHA_SOFT_RAD = radians(22.0)  # self-imposed AoA soft cap (below the FLCS 30° hard limiter)
BREAK_BANK_RAD = radians(75.0)  # commanded bank for a hard defensive break turn
BREAK_SWEEP_RAD = radians(150.0)  # heading swept before a break turn recovers
SPLITS_SWEEP_RAD = radians(165.0)  # ground-course reversal before a Split-S recovers
MANEUVER_MAX_S = 12.0  # hard safety cap on any single maneuver's duration
RECOVER_S = 1.0  # wings-level rollout after a maneuver before returning to hold
MANEUVER_COOLDOWN_S = 3.0  # min dwell in level-hold before a maneuver may retrigger (anti-chatter)

# --- Tier-3 FSM tactician: BFM thresholds + WEZ trigger (§8.5) ---
# All transitions are dead-banded (separate enter/exit thresholds) and dwell-gated.
FIRE_LEAD_RAD = radians(4.0)  # fire when the sim's aim error < this and its solution is in range
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
V_ENERGY_LOW = 120.0  # m/s; airspeed floor in a break ⇒ energy recovery
V_ENERGY_OK = 175.0  # m/s (~corner); recovered ⇒ re-engage
V_PURSUIT = 200.0  # m/s; target speed the pursuit throttle law holds
PURSUE_ROLL_DEADBAND = radians(5.0)  # stop rolling once the target is this near boresight
K_PURSUE_ROLL = 2.0  # lift-vector-on-target roll gain
K_PURSUE_PULL = 1.5  # pull gain (× off-boresight angle) once rolled on
FSM_MIN_DWELL_S = 0.4  # min time in a state before a non-survival transition (anti-chatter)
# Tier-3's own raw control laws (not the sim autopilot's gains).
K_ROLL_DAMP = 0.4  # aileron per rad/s of roll rate
K_BREAK_ROLL = 2.5  # aileron per rad of bank error in a break
PURSUIT_THR_TRIM = 0.6  # throttle at V_PURSUIT
K_PURSUIT_THR = 0.02  # throttle per m/s of airspeed error

# --- reward magnitudes (§8.3) ---
R_KILL = 100.0  # terminal: bandit destroyed
R_DEATH = -100.0  # terminal: learner destroyed
R_DECK = -200.0  # terminal: hard-deck bust (≥ death — the worst outcome)
R_BUBBLE_EXIT = -50.0  # truncate: left the engagement bubble
DMG_PER_HP = 0.5  # dense: ±0.5 per HP dealt / taken (a 2-HP hit = +1.0)
K_GEOM = 0.08  # dense: potential-based geometry shaping scale
TIME_PENALTY = -0.001  # dense: per-step time cost
GAMMA = 0.99  # discount used in the shaping term γΦ(s') − Φ(s)
