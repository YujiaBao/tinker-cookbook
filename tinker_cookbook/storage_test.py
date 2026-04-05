"""Tests for the Storage abstraction layer."""

import json
from pathlib import Path

import pytest

from tinker_cookbook.storage import (
    LocalStorage,
    Storage,
    StorageStat,
    storage_append_jsonl,
    storage_read_json,
    storage_read_jsonl,
    storage_write_json,
)


class TestLocalStorage:
    def test_write_and_read(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("test.txt", b"hello world")
        assert storage.read("test.txt") == b"hello world"

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("a/b/c/file.txt", b"deep")
        assert storage.read("a/b/c/file.txt") == b"deep"

    def test_write_overwrites(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("f.txt", b"v1")
        storage.write("f.txt", b"v2")
        assert storage.read("f.txt") == b"v2"

    def test_read_missing_raises(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        with pytest.raises(FileNotFoundError):
            storage.read("nonexistent.txt")

    def test_append(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.append("log.txt", b"line1\n")
        storage.append("log.txt", b"line2\n")
        assert storage.read("log.txt") == b"line1\nline2\n"

    def test_append_creates_file(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.append("new.txt", b"first")
        assert storage.read("new.txt") == b"first"

    def test_exists(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert not storage.exists("nope.txt")
        storage.write("yes.txt", b"hi")
        assert storage.exists("yes.txt")

    def test_stat(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert storage.stat("nope.txt") is None
        storage.write("f.txt", b"12345")
        stat = storage.stat("f.txt")
        assert stat is not None
        assert stat.size == 5
        assert stat.mtime > 0

    def test_list_dir(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("dir/a.txt", b"")
        storage.write("dir/b.txt", b"")
        storage.write("dir/sub/c.txt", b"")
        items = storage.list_dir("dir")
        assert "a.txt" in items
        assert "b.txt" in items
        assert "sub" in items

    def test_list_dir_empty(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert storage.list_dir("nonexistent") == []

    def test_implements_protocol(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert isinstance(storage, Storage)


class TestStorageJsonHelpers:
    def test_read_json(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        data = {"key": "value", "num": 42}
        storage.write("test.json", json.dumps(data).encode())
        result = storage_read_json(storage, "test.json")
        assert result == data

    def test_read_json_missing(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert storage_read_json(storage, "nope.json") is None

    def test_read_json_malformed(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("bad.json", b"not json")
        assert storage_read_json(storage, "bad.json") is None

    def test_read_jsonl(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        lines = [json.dumps({"step": i}) for i in range(3)]
        storage.write("test.jsonl", ("\n".join(lines) + "\n").encode())
        result = storage_read_jsonl(storage, "test.jsonl")
        assert len(result) == 3
        assert result[0]["step"] == 0
        assert result[2]["step"] == 2

    def test_read_jsonl_missing(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert storage_read_jsonl(storage, "nope.jsonl") == []

    def test_read_jsonl_skips_malformed(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("mixed.jsonl", b'{"ok": 1}\nnot json\n{"ok": 2}\n')
        result = storage_read_jsonl(storage, "mixed.jsonl")
        assert len(result) == 2

    def test_write_json(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage_write_json(storage, "out.json", {"x": 1})
        result = storage_read_json(storage, "out.json")
        assert result == {"x": 1}

    def test_append_jsonl(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage_append_jsonl(storage, "out.jsonl", {"step": 0})
        storage_append_jsonl(storage, "out.jsonl", {"step": 1})
        result = storage_read_jsonl(storage, "out.jsonl")
        assert len(result) == 2
        assert result[0]["step"] == 0
        assert result[1]["step"] == 1
