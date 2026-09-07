import sys, time
sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa
import astrum_env
from astrum_dogfight.scenario import tier1_scenario

# warm one build (first FDM load may be cold)
toml = tier1_scenario(separation_m=500.0)
t0=time.perf_counter(); e=astrum_env.DogfightEnv(toml); t1=time.perf_counter()
print(f"COLD build: {(t1-t0)*1e3:.1f} ms")
e.reset()
# time N warm builds
N=10
t0=time.perf_counter()
for i in range(N):
    ee=astrum_env.DogfightEnv(tier1_scenario(separation_m=400.0+i*10))
    ee.reset()
t1=time.perf_counter()
print(f"WARM build+reset avg over {N}: {(t1-t0)/N*1e3:.1f} ms")
# time N resets on one env
t0=time.perf_counter()
for i in range(N):
    e.reset()
t1=time.perf_counter()
print(f"reset() avg over {N}: {(t1-t0)/N*1e3:.1f} ms")
