"""Phase identity and portable asset provenance, independent of the GUI.

Numeric IDs are file-format identifiers. Keys identify phases across renames,
reordering and workflow reloads; names are deliberately never used as identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from uuid import uuid4


COLORS = ("#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb")


def file_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_fingerprint(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DictionaryProvenance:
    master_sha256: str
    structure_sha256: str
    settings_json: str
    schema: int = 1

    @classmethod
    def create(cls, master_sha256: str, structure: dict, settings: dict):
        return cls(master_sha256, metadata_fingerprint(structure),
                   json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def compatible_with(self, other: DictionaryProvenance) -> bool:
        return bool(self.master_sha256) and self == other

    def differs_only_in_pc(self, other: DictionaryProvenance) -> bool:
        if (not self.master_sha256 or self.master_sha256 != other.master_sha256
                or self.structure_sha256 != other.structure_sha256 or self.schema != other.schema):
            return False
        saved, current = json.loads(self.settings_json), json.loads(other.settings_json)
        saved_pc, current_pc = saved.pop("pc_bruker", None), current.pop("pc_bruker", None)
        return saved_pc is not None and current_pc is not None and saved_pc != current_pc and saved == current


@dataclass
class DictionaryAsset:
    key: str = field(default_factory=lambda: uuid4().hex)
    path: str = ""
    provenance: DictionaryProvenance | None = None
    # Explicit linking permits legacy assets, but never calls them verified.
    legacy_linked: bool = False
    persistent: bool = True
    accepted_pc_bruker: list[float] | None = None

    def compatible_with(self, expected: DictionaryProvenance) -> bool:
        if self.provenance is None:
            return False
        return self.provenance.compatible_with(expected) or (
            self.accepted_pc_bruker is not None
            and self.accepted_pc_bruker == json.loads(expected.settings_json).get("pc_bruker")
            and self.provenance.differs_only_in_pc(expected)
        )


@dataclass
class PhaseEntry:
    output_id: int
    name: str
    key: str = field(default_factory=lambda: uuid4().hex)
    color: str = COLORS[0]
    enabled: bool = True
    input_ids: list[int] = field(default_factory=list)
    structure: dict = field(default_factory=dict)
    master_path: str = ""
    master_sha256: str = ""
    dictionaries: list[DictionaryAsset] = field(default_factory=list)
    active_dictionary_key: str | None = None

    @property
    def active_dictionary(self):
        return next((d for d in self.dictionaries if d.key == self.active_dictionary_key), None)

    def status(self, expected: DictionaryProvenance | None = None) -> str:
        if not self.master_path or not Path(self.master_path).is_file():
            return "Master missing"
        asset = self.active_dictionary
        if asset is None or not asset.path or not Path(asset.path).is_file():
            return "Dictionary missing"
        if asset.provenance is None:
            return "Legacy / unverified"
        if asset.provenance.master_sha256 != self.master_sha256:
            return "Incompatible"
        if expected is not None and not asset.compatible_with(expected):
            return "Incompatible"
        return "Ready" if expected is not None else "Check settings"


@dataclass(frozen=True)
class PhaseRunBinding:
    key: str
    output_id: int
    master_path: str
    master_sha256: str
    dictionary_path: str
    provenance: DictionaryProvenance


class PhaseRegistry:
    SCHEMA = 1

    def __init__(self):
        self.entries: list[PhaseEntry] = []
        self.search_revision = 0

    def add(self, name: str, *, output_id: int | None = None,
            input_ids=(), structure: dict | None = None) -> PhaseEntry:
        used = {e.output_id for e in self.entries}
        if output_id is None:
            output_id = max(used, default=0) + 1
        if output_id <= 0 or output_id in used:
            raise ValueError("Phase output IDs must be unique positive integers.")
        if any(set(input_ids).intersection(e.input_ids) for e in self.entries):
            raise ValueError("An input phase ID is already linked.")
        entry = PhaseEntry(output_id=output_id, name=name, input_ids=list(input_ids),
                           structure=dict(structure or {}), color=COLORS[len(self.entries) % len(COLORS)])
        self.entries.append(entry)
        self.search_revision += 1
        return entry

    def by_key(self, key: str) -> PhaseEntry:
        return next(e for e in self.entries if e.key == key)

    def by_output_id(self, phase_id: int) -> PhaseEntry | None:
        return next((e for e in self.entries if e.output_id == phase_id), None)

    def set_enabled(self, key: str, enabled: bool):
        entry = self.by_key(key)
        if entry.enabled != bool(enabled):
            entry.enabled = bool(enabled)
            self.search_revision += 1

    def bind_master(self, key: str, path: str, *, expected_sha256: str | None = None):
        entry = self.by_key(key)
        path = str(Path(path).expanduser().resolve())
        fingerprint = file_fingerprint(path)
        if expected_sha256 and fingerprint != expected_sha256:
            raise ValueError("The selected master does not match the saved master fingerprint.")
        if entry.master_sha256 != fingerprint:
            self.search_revision += 1
        entry.master_path, entry.master_sha256 = path, fingerprint

    def snapshot(self, expected: dict[str, DictionaryProvenance]) -> tuple[PhaseRunBinding, ...]:
        bindings = []
        for entry in sorted(self.entries, key=lambda e: e.output_id):
            if not entry.enabled:
                continue
            provenance = expected.get(entry.key)
            status = entry.status(provenance)
            if provenance is None or status != "Ready":
                raise ValueError(f"{entry.name}: {status}.")
            bindings.append(PhaseRunBinding(entry.key, entry.output_id, entry.master_path,
                                            entry.master_sha256, entry.active_dictionary.path,
                                            entry.active_dictionary.provenance))
        if not bindings:
            raise ValueError("Enable at least one phase.")
        return tuple(bindings)

    def to_json(self) -> str:
        return json.dumps(dict(schema=self.SCHEMA, search_revision=self.search_revision,
                               phases=[asdict(e) for e in self.entries]), allow_nan=False)

    @classmethod
    def from_json(cls, value: str):
        payload = json.loads(value)
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError("Unsupported phase registry schema.")
        registry = cls()
        keys, output_ids, input_ids = set(), set(), set()
        for fields in payload["phases"]:
            fields = dict(fields)
            assets = []
            for asset in fields.pop("dictionaries", []):
                asset = dict(asset)
                if asset.get("provenance") is not None:
                    asset["provenance"] = DictionaryProvenance(**asset["provenance"])
                assets.append(DictionaryAsset(**asset))
            entry = PhaseEntry(**fields, dictionaries=assets)
            if not entry.key or entry.key in keys or entry.output_id <= 0 or entry.output_id in output_ids:
                raise ValueError("Invalid or duplicate phase identity in workflow.")
            if input_ids.intersection(entry.input_ids) or len(set(entry.input_ids)) != len(entry.input_ids):
                raise ValueError("Ambiguous input phase mapping in workflow.")
            if len({a.key for a in assets}) != len(assets):
                raise ValueError("Duplicate dictionary identity in workflow.")
            if entry.active_dictionary_key is not None and entry.active_dictionary is None:
                raise ValueError("Active dictionary is absent from phase assets.")
            keys.add(entry.key)
            output_ids.add(entry.output_id)
            input_ids.update(entry.input_ids)
            registry.entries.append(entry)
        registry.search_revision = int(payload.get("search_revision", 0))
        return registry
