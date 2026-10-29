# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("restore", Path(__file__).with_name("restore_snapshot.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_restore_exact_bytes_modes_and_refuse_overwrite(tmp_path):
    data = b"historical raw evidence\x00\xff"
    sha = M.blob_sha(data)
    snapshot = {"branch": "old", "head": "frozen", "files": {"src/tool": {"mode": "100755", "sha": sha}}}
    out = tmp_path / "restored"
    M.restore(snapshot, out, lambda _: data)
    assert (out / "src/tool").read_bytes() == data
    assert (out / "src/tool").stat().st_mode & 0o111
    with pytest.raises(FileExistsError):
        M.restore(snapshot, out, lambda _: data)


def test_restore_rejects_bad_paths_or_objects(tmp_path):
    for path in ("../escape", "/absolute"):
        with pytest.raises(ValueError):
            M.restore({"files": {path: {}}}, tmp_path / "new", lambda _: b"")
    with pytest.raises(ValueError):
        M.restore({"files": {"file": {"sha": "wrong", "mode": "100644"}}}, tmp_path / "corrupt", lambda _: b"x")
