import sys, time, psutil, os
sys.path.insert(0, ".")
from src.evaluation.pipeline import build_entries
ds, seed = sys.argv[1], int(sys.argv[2])
proc = psutil.Process(os.getpid()); t0=time.time()
e = build_entries([ds], [seed], log=lambda m: print(m, flush=True))[0]
r = e["risks"]
print(f"OK {e['stream_id']}: {len(r)} win eps={e['eps']:.4f} "
      f"risk[{r.min():.3f},{r.max():.3f}] valid={int((r<=e['eps']).sum())} "
      f"peakRSS={proc.memory_info().rss/1e9:.1f}GB {time.time()-t0:.0f}s", flush=True)
