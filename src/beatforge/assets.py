"""Provenance-first local asset catalog for future browser/Vivify import."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


SAFE_LICENSES = frozenset({"CC0", "CC-BY"})


@dataclass(frozen=True, slots=True)
class AssetRecord:
    asset_id: str
    title: str
    source_url: str
    local_path: str
    license: str
    author: str = ""
    sha256: str = ""
    media_type: str = "model"

    def with_hash(self) -> "AssetRecord":
        digest = hashlib.sha256(Path(self.local_path).read_bytes()).hexdigest()
        return AssetRecord(self.asset_id, self.title, self.source_url, self.local_path,
                           self.license, self.author, digest, self.media_type)


class AssetLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, AssetRecord] = {}
        if self.path.exists():
            for raw in json.loads(self.path.read_text(encoding="utf-8")):
                record = AssetRecord(**raw)
                self._records[record.asset_id] = record

    def add(self, record: AssetRecord, *, require_local_file: bool = True) -> AssetRecord:
        if record.license.upper() not in SAFE_LICENSES:
            raise ValueError(f"asset license is not in the safe default allowlist: {record.license!r}")
        local = Path(record.local_path)
        if require_local_file and not local.is_file():
            raise FileNotFoundError(local)
        if local.is_file():
            record = record.with_hash()
        self._records[record.asset_id] = record
        self._save()
        return record

    def get(self, asset_id: str) -> AssetRecord | None:
        return self._records.get(asset_id)

    def search(self, text: str, media_type: str | None = None) -> list[AssetRecord]:
        needle = text.casefold()
        return [record for record in self._records.values()
                if (not needle or needle in record.title.casefold() or needle in record.author.casefold())
                and (media_type is None or record.media_type == media_type)]

    def records(self) -> Iterable[AssetRecord]:
        return tuple(self._records.values())

    def _save(self) -> None:
        payload = [asdict(record) for record in sorted(self._records.values(), key=lambda item: item.asset_id)]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
