"""Build and cache monitored streams for every (dataset, seed)."""
import sys, time
sys.path.insert(0, ".")
from src.evaluation.pipeline import build_entries
from src.utils.seeding import SEEDS_FULL

t0 = time.time()
entries = build_entries(["elec2", "insects", "yearbook"], list(SEEDS_FULL), log=print)
print(f"\nDONE {len(entries)} streams in {time.time()-t0:.0f}s")
for e in entries:
    r = e["risks"]
    print(f"  {e['stream_id']:22s} fam={e['family']:16s} win={len(r):4d} "
          f"eps={e['eps']:.4f} risk[{r.min():.3f},{r.max():.3f}] "
          f"valid={int((r<=e['eps']).sum())}")
