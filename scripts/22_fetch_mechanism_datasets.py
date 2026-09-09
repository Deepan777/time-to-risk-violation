#!/usr/bin/env python3
"""22_fetch_mechanism_datasets.py -- download the PR-H1 archives and record what was downloaded.

The drift class of every dataset here was fixed in `configs/drift_mechanism.yaml` and committed
*before* this script was first run, so that no class could be assigned after seeing data. This
script therefore does not decide anything; it fetches, hashes, and writes a provenance file. If a
future re-run produces a different SHA-256 the archive changed upstream, and the results in the
manuscript no longer describe the file that is there now -- which is why the hash is recorded
rather than merely the URL.
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.cli import standard_parser                            # noqa: E402
from src.utils.io import ensure_dir, get_logger, write_json          # noqa: E402

OUT = Path("data/raw/uci")
PROVENANCE = Path("results/processed/mechanism_downloads.json")

#: name -> (UCI dataset id, archive filename on the UCI static host)
ARCHIVES: dict[str, tuple[str, str]] = {
    "gas_drift":       ("224", "gas+sensor+array+drift+dataset.zip"),
    "air_quality":     ("360", "air+quality.zip"),
    "gas_temp_mod":    ("487", "gas+sensor+array+temperature+modulation.zip"),
    "news_popularity": ("332", "online+news+popularity.zip"),
    "hydraulic":       ("447", "condition+monitoring+of+hydraulic+systems.zip"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = standard_parser("Fetch the PR-H1 drift-mechanism archives", require_config=False)
    args = ap.parse_args()
    ensure_dir(OUT)
    ensure_dir(PROVENANCE.parent)
    log = get_logger("fetch_mechanism", args.log or "logs/fetch_mechanism.log").info

    records = {}
    for name, (uci_id, fn) in ARCHIVES.items():
        dst = OUT / f"{name}.zip"
        url = f"https://archive.ics.uci.edu/static/public/{uci_id}/{fn}"
        if dst.is_file() and not args.force:
            log(f"{name}: present, not re-downloaded")
        else:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            log(f"{name}: {dst.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s")
        records[name] = {
            "uci_id": uci_id, "url": url, "path": str(dst),
            "bytes": int(dst.stat().st_size), "sha256": sha256(dst),
        }
        log(f"  sha256={records[name]['sha256']}")

    write_json({"archives": records,
                "note": "Classes were fixed in configs/drift_mechanism.yaml before this ran."},
               PROVENANCE)
    log(f"provenance written to {PROVENANCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
