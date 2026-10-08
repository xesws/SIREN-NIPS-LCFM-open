#!/usr/bin/env python3
"""Download the versioned frozen evidence bundle, verify, and safely unpack it."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("unsafe archive member")
            if not path.parts or path.parts[0] != "frozen":
                raise ValueError("unexpected archive root")
        for member in source.getmembers():
            target = destination.joinpath(*PurePosixPath(member.name).parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/download.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "artifacts/frozen")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive (still hash-checked)")
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    if args.destination.exists() and any(args.destination.iterdir()):
        # A complete matching payload can be reused; never overwrite a different run.
        local = args.destination / "manifest.json"
        if local.is_file() and sha256(local) == spec["payload_manifest_sha256"]:
            entries = json.loads(local.read_text())["files"]
            for name, entry in entries.items():
                if name.startswith("artifacts/frozen/"):
                    path = args.destination / Path(name).relative_to("artifacts/frozen")
                    if not path.is_file() or sha256(path) != entry["sha256"]:
                        raise ValueError("existing payload is incomplete or changed; choose a new --destination")
            print(json.dumps({"reused": True, "destination": str(args.destination)}))
            return
        raise ValueError("destination already contains different files; choose an empty --destination")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="siren-evidence-", dir=args.destination.parent) as work:
        work = Path(work)
        archive = args.archive or work / spec["filename"]
        if args.archive is None:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "siren-rule-memory/0.1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
        if archive.stat().st_size != spec["bytes"] or sha256(archive) != spec["sha256"]:
            raise ValueError("archive size or SHA-256 mismatch")
        staged = work / "unpacked"
        unpack(archive, staged)
        if sha256(staged / "manifest.json") != spec["payload_manifest_sha256"]:
            raise ValueError("payload manifest mismatch")
        if args.destination.exists():
            args.destination.rmdir()  # Empty destination only, checked above.
        os.replace(staged, args.destination)
    print(json.dumps({"downloaded": True, "destination": str(args.destination), "sha256": spec["sha256"]}))


if __name__ == "__main__":
    main()
