"""Apply the curriculum-87 obs-widening surgery to a checkpoint and save it, so a PRE-append
policy can be scored in the POST-append env. (throwaway tool)

Why this is needed at all. curriculum-86 changed two observation columns as a side effect of a
reward fix, and the c84-vs-c86 comparison was corrupted by it: c84 read 2 kills against its
recorded 10 purely because it was handed out-of-distribution inputs
([[curriculum-86-corrected-aim]]). curriculum-87 widens the obs 31 -> 41, which would repeat the
mistake in a louder form — a hard shape error instead of a silent misread. `_widen_resume_obs`
already zero-pads the encoder so the resumed policy is bit-identical on the trained features; this
script exposes that as a standalone transform so the widened c86 can be verdicted directly.

That serves two purposes at once:
  1. It VALIDATES the bit-identity claim behaviourally. The widened c86 should reproduce c86's
     recorded score (22 kills / 1 loss / NET +28.8 / 32%). If it does not, the new columns are not
     silent and the warm start is not the controlled experiment it is claimed to be.
  2. It IS the control for c87. Scored this way, baseline and candidate are measured by the same
     harness in the same env, which is exactly what the c86 comparison could not do.

Usage:  uv run python _widen_ckpt_tmp.py <in.pt> <out.pt> [frame_stack]
"""

from __future__ import annotations

import sys

import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2.train_parallel import _widen_resume_obs


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    frame_stack = int(sys.argv[3]) if len(sys.argv) == 4 else 1

    _paths.require_astrum_env()
    from astrum_dogfight import constants as C

    sd = torch.load(src, map_location="cpu", weights_only=False)
    ckpt_obs = int(sd["cfg"]["obs_dim"])
    want = C.OBS_DIM * frame_stack
    if ckpt_obs == want:
        print(f"[widen] {src} is already {want}-wide — nothing to do.")
        return 0
    _widen_resume_obs(sd, ckpt_obs // frame_stack, C.OBS_DIM, frame_stack)
    torch.save(sd, dst)
    print(
        f"[widen] wrote {dst} (obs {ckpt_obs} -> {sd['cfg']['obs_dim']}, "
        f"collected={sd.get('collected', '?')}, stage_idx={sd.get('stage_idx', '?')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
