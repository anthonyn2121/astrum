import time
import _diag_d3b_express as E

t0 = time.time()
r, rows = E.fly(E.Sched([(24.0, 0.0, 1.0, 1.0)]), n_steps=int(round(8.0 / E.DT)))
t1 = time.time()
print(f"8s rollout: {r.steps} steps in {t1-t0:.2f} s wall -> {(t1-t0)/max(1,r.steps)*1000:.2f} ms/step")
t0 = time.time()
r, rows = E.fly(E.Sched([(24.0, 0.0, 1.0, 1.0)]))
t1 = time.time()
print(f"24s rollout: {r.steps} steps in {t1-t0:.2f} s wall")
