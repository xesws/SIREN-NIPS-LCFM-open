#!/usr/bin/env python3
"""Assemble the public frozen payload without weights, caches or private logs."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0.1.0")
    args = parser.parse_args()
    source = ROOT / "artifacts/frozen"
    index = json.loads((source / "manifest.json").read_text())
    allowed = {str(Path(p).relative_to("artifacts/frozen")) for p in index["files"] if p.startswith("artifacts/frozen/")} | {"manifest.json"}
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    if actual != allowed:
        raise ValueError(f"unexpected or missing files: {sorted(actual ^ allowed)}")
    for name in sorted(actual):
        path = source / name
        if path.is_symlink() or path.suffix in (".pt", ".pth", ".safetensors"):
            raise ValueError("weights/symlinks are not part of this release")
        if name != "manifest.json" and sha(path) != index["files"][f"artifacts/frozen/{name}"]["sha256"]:
            raise ValueError("frozen export changed")
    target = ROOT / "artifacts/packages"
    target.mkdir(parents=True, exist_ok=True)
    filename = f"siren-paper-evidence-{args.tag}.tar.gz"
    archive = target / filename
    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped, tarfile.open(fileobj=zipped, mode="w") as tar:
        for name in sorted(actual):
            p = source / name
            info = tar.gettarinfo(str(p), arcname="frozen/" + name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with p.open("rb") as f:
                tar.addfile(info, f)
    checksum = sha(archive)
    (target / (filename + ".sha256")).write_text(f"{checksum}  {filename}\n")
    manifest = {"schema": "siren-public-download-v1", "tag": args.tag, "filename": filename,
                "url": f"https://github.com/xesws/SIREN-NIPS-LCFM-open/releases/download/{args.tag}/{filename}",
                "bytes": archive.stat().st_size, "sha256": checksum,
                "payload_manifest_sha256": sha(source / "manifest.json"), "files": len(actual),
                "contains_model_weights": False, "contains_teacher_probability_cache": False}
    (ROOT / "artifacts/download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
