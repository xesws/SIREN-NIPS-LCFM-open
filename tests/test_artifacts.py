import importlib.util
import io
from pathlib import Path
import tarfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("artifact_download", ROOT / "reproduce/download_artifacts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_archive_cannot_write_outside_destination(tmp_path):
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("frozen/../../outside")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        module.unpack(path, tmp_path / "target")
    assert not (tmp_path / "outside").exists()


def test_archive_accepts_regular_evidence_files(tmp_path):
    path = tmp_path / "good.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("frozen/routing/sample.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    module.unpack(path, tmp_path / "target")
    assert (tmp_path / "target/routing/sample.json").read_text() == "{}"
