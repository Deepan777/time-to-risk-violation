import gdown, sys
from pathlib import Path
out = Path("data/raw/wildtime/huffpost.pkl")
if out.exists() and out.stat().st_size > 0:
    print(f"present: {out.stat().st_size/1e6:.1f} MB"); sys.exit(0)
gdown.download(url="https://drive.google.com/u/0/uc?id=1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY&export=download",
               output=str(out), quiet=True)
print(f"downloaded {out.stat().st_size/1e6:.1f} MB")
