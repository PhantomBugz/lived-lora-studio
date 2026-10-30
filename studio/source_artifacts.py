"""Fail-closed validation for reviewed source-code artifacts."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
import textwrap
import tomllib
from types import MappingProxyType
from typing import Literal
from xml.etree import ElementTree


ALLOWED_EXTENSIONS = frozenset(
    {
        ".txt", ".md", ".rst", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh",
        ".hpp", ".js", ".ts", ".py", ".rs", ".go", ".java", ".cs", ".inf",
        ".inx", ".vcxproj", ".props", ".targets", ".sln", ".cmake", ".json",
        ".yaml", ".yml", ".toml", ".xml",
    }
)
MAX_ARTIFACT_BYTES = 512 * 1024
MAX_PATH_BYTES = 1024
MAX_PATH_PARTS = 64
MAX_SEGMENT_BYTES = 4096
REVIEW_CANDIDATE_SCORE = 55
TECHNICAL_SIGNALS = (
    "definitions_types", "api_calls", "control_data_flow", "technical_explanation",
    "concrete_procedure", "commands", "configuration_schema", "error_handling",
    "tests", "examples",
)
_SIGNAL_WEIGHTS = {
    "definitions_types": 28, "api_calls": 24, "control_data_flow": 25,
    "technical_explanation": 22, "concrete_procedure": 20, "commands": 24,
    "configuration_schema": 22, "error_handling": 14, "tests": 12, "examples": 12,
}
_C_FAMILY_EXTENSIONS = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".java", ".cs"})
_PROGRAMMING_EXTENSIONS = _C_FAMILY_EXTENSIONS | {".py", ".js", ".ts", ".rs", ".go"}
_XML_EXTENSIONS = frozenset({".xml", ".vcxproj", ".props", ".targets"})

_BINARY_SIGNATURES = (
    b"MZ", b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe", b"PK\x03\x04", b"\x1f\x8b",
    b"%PDF-", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    b"SQLite format 3\x00", b"L\x00\x00\x00\x01\x14\x02\x00", b"GIF87a", b"GIF89a",
)
_DEPENDENCY_PARTS = frozenset({"__pycache__", "node_modules", "vendor", "dist", "build"})
_LOCKFILE_NAMES = frozenset(
    {"package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "poetry.lock", "Pipfile.lock"}
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"https?://[^\s/:@]+:[^\s@]+@", re.IGNORECASE),
)
_CREDENTIAL_NAME = r"(?:api[_-]?key|token|secret|password|access[_-]?key|client[_-]?secret)"
_QUOTED_CREDENTIAL_VALUE = r'''(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')'''
_CREDENTIAL_ASSIGNMENT = re.compile(
    rf"""(?i)(?<![\w-])(?P<quote>['"]?){_CREDENTIAL_NAME}(?P=quote)(?![\w-])"""
    r"(?:[ \t]*\[[ \t]*\])?"
    r"(?:[ \t]*:[ \t]*&?[A-Za-z_][\w.\[\], |<>?]*?|[ \t]+string)?[ \t]*"
    rf"(?P<operator>:=|=(?!=)|:)\s*(?P<value>{_QUOTED_CREDENTIAL_VALUE}|[^\s#;,}}\]]+)"
    r"(?=[ \t]*(?:$|[\r\n;,#}\]]|//))"
    # Consume unrelated string literals whole so embedded assignment-like text
    # cannot be mistaken for a declaration or data key.
    rf"|{_QUOTED_CREDENTIAL_VALUE}"
)
_CREDENTIAL_JSON_VALUE = re.compile(
    rf'(?i)"{_CREDENTIAL_NAME}"\s*:\s*(?P<value>"[^"\n]*")'
)
_CREDENTIAL_XML_VALUE = re.compile(
    rf"(?i)<(?P<name>{_CREDENTIAL_NAME})\s*>\s*(?P<value>[^<\s]+)\s*</(?P=name)\s*>"
)
_CREDENTIAL_PLACEHOLDERS = frozenset(
    {"REPLACE_ME", "YOUR_API_KEY", "YOUR_TOKEN", "CHANGEME", "EXAMPLE", "PLACEHOLDER", "NONE", "NULL"}
)


class ArtifactRejected(ValueError):
    """A source artifact failed a stable, non-sensitive validation rule."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


class FrozenJsonDict(Mapping[str, object]):
    """A recursively immutable JSON-compatible metadata mapping."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True, slots=True)
class ArtifactInput:
    source_kind: Literal["github_blob", "forum_attachment"]
    path: str
    raw_bytes: bytes
    provenance: Mapping[str, object]
    provenance_strength: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", freeze_json_metadata(self.provenance))


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    path: str
    text: str
    raw_sha256: str
    canonical_text_sha256: str
    provenance: FrozenJsonDict
    provenance_strength: int


@dataclass(frozen=True, slots=True)
class TechnicalAssessment:
    score: int
    signals: tuple[str, ...]
    disposition: Literal["review_candidate", "quarantined"]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactSegment:
    text: str
    index: int
    count: int
    score: int
    signals: tuple[str, ...]
    disposition: Literal["review_candidate", "quarantined"]
    reasons: tuple[str, ...]
    raw_sha256: str
    canonical_text_sha256: str
    fingerprints: frozenset[int]


@dataclass(frozen=True, slots=True)
class DedupeRecord:
    segment: ArtifactSegment
    path: str
    provenance_hash: str
    provenance_strength: int


@dataclass(frozen=True, slots=True)
class DedupeDecision:
    provenance_hash: str
    winner_provenance_hash: str
    disposition: Literal["winner", "duplicate"]
    reason: str | None


def validate_artifact(value: ArtifactInput) -> ValidatedArtifact:
    """Validate an untrusted artifact before it enters the review queue."""

    if not isinstance(value, ArtifactInput):
        raise ArtifactRejected("invalid_artifact_input")
    path = normalize_artifact_path(value.path)
    if value.source_kind not in {"github_blob", "forum_attachment"}:
        raise ArtifactRejected("invalid_source_kind")
    require_allowed_extension(path)
    require_size(value.raw_bytes, maximum=MAX_ARTIFACT_BYTES)
    reject_binary_signature(value.raw_bytes)
    reject_lfs_pointer(value.raw_bytes)
    text = decode_strict_utf8(value.raw_bytes)
    reject_generated_or_dependency(path, text)
    reject_secret_material(text, path)
    canonical = canonicalize_text(text)
    return ValidatedArtifact(
        path=path,
        text=text,
        raw_sha256=sha256(value.raw_bytes).hexdigest(),
        canonical_text_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
        provenance=freeze_json_metadata(value.provenance),
        provenance_strength=validate_provenance_strength(value.provenance_strength),
    )


def normalize_artifact_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactRejected("unsafe_path")
    path_bytes = _encode_path_utf8(value)
    if path_bytes is None or len(path_bytes) > MAX_PATH_BYTES:
        raise ArtifactRejected("unsafe_path")
    if value.startswith("/") or value.startswith("\\") or re.match(r"^[A-Za-z]:", value):
        raise ArtifactRejected("unsafe_path")
    if "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ArtifactRejected("unsafe_path")
    parts = value.split("/")
    if len(parts) > MAX_PATH_PARTS or any(part in {"", ".", ".."} for part in parts):
        raise ArtifactRejected("unsafe_path")
    return value


def _encode_path_utf8(value: str) -> bytes | None:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        return None


def require_allowed_extension(path: str) -> None:
    lowered = path.lower()
    extension = "." + lowered.rsplit(".", 1)[1] if "." in lowered.rsplit("/", 1)[-1] else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise ArtifactRejected("unsupported_extension")


def require_size(raw_bytes: bytes, *, maximum: int) -> None:
    if not isinstance(raw_bytes, bytes):
        raise ArtifactRejected("invalid_artifact_bytes")
    if len(raw_bytes) > maximum:
        raise ArtifactRejected("artifact_too_large")


def reject_binary_signature(raw_bytes: bytes) -> None:
    if any(raw_bytes.startswith(signature) for signature in _BINARY_SIGNATURES):
        raise ArtifactRejected("binary_signature")


def reject_lfs_pointer(raw_bytes: bytes) -> None:
    if raw_bytes.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        raise ArtifactRejected("git_lfs_pointer")


def decode_strict_utf8(raw_bytes: bytes) -> str:
    if b"\x00" in raw_bytes:
        raise ArtifactRejected("nul_byte")
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = None
    if text is None:
        raise ArtifactRejected("invalid_utf8")
    if canonicalize_text(text).strip() == "":
        raise ArtifactRejected("empty_text")
    return text


def reject_generated_or_dependency(path: str, text: str) -> None:
    parts = path.lower().split("/")
    filename = parts[-1]
    if (
        any(part in _DEPENDENCY_PARTS for part in parts)
        or filename in {name.lower() for name in _LOCKFILE_NAMES}
        or filename.endswith(".min.js")
        or filename.endswith(".map")
        or re.search(r"(?:^|\n)\s*(?:@generated|generated by|do not edit)\b", text, re.IGNORECASE)
        or _looks_minified(text)
    ):
        raise ArtifactRejected("generated_or_dependency")


def _looks_minified(text: str) -> bool:
    punctuation = sum(not char.isalnum() and not char.isspace() for char in text)
    return (
        len(text) > 2_000
        and "\n" not in text
        and sum(char.isspace() for char in text) < len(text) // 50
        and punctuation > len(text) // 4
    )


def reject_secret_material(text: str, path: str) -> None:
    suffix = "." + path.lower().rsplit(".", 1)[-1]
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS) or _has_literal_credential_assignment(text, suffix):
        raise ArtifactRejected("secret_material")


def _has_literal_credential_assignment(text: str, suffix: str) -> bool:
    for match in _CREDENTIAL_ASSIGNMENT.finditer(text):
        if match.group("value") is None:
            continue
        if _is_literal_credential(
            match.group("value"),
            allow_reference=(
                suffix in _PROGRAMMING_EXTENSIONS
                and match.group("operator") in {"=", ":="}
                and not match.group("quote")
            ),
        ):
            return True
    data_pattern = (
        _CREDENTIAL_JSON_VALUE if suffix == ".json" else
        _CREDENTIAL_XML_VALUE if suffix in _XML_EXTENSIONS else None
    )
    if data_pattern is not None:
        return any(
            _is_literal_credential(match.group("value"), allow_reference=False)
            for match in data_pattern.finditer(text)
        )
    return False


def _is_literal_credential(value: str, *, allow_reference: bool) -> bool:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        literal = value[1:-1]
        normalized = literal.upper()
        if normalized in _CREDENTIAL_PLACEHOLDERS or any(
            marker in normalized for marker in ("REPLACE", "PLACEHOLDER", "YOUR_")
        ):
            return False
        return bool(literal)
    if allow_reference and "_" in value and value.isidentifier():
        return False
    return bool(
        re.fullmatch(r"[A-Za-z0-9_-]{32,}", value)
        and any(character.isalpha() for character in value)
        and any(character.isdigit() for character in value)
    )


def canonicalize_text(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))


def freeze_json_metadata(value: Mapping[str, object]) -> FrozenJsonDict:
    if not isinstance(value, Mapping):
        raise ArtifactRejected("invalid_provenance")
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ArtifactRejected("invalid_provenance")
        frozen[key] = _freeze_json_value(item)
    return FrozenJsonDict(frozen)


def _freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ArtifactRejected("invalid_provenance")
    if isinstance(value, Mapping):
        return freeze_json_metadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise ArtifactRejected("invalid_provenance")


def validate_provenance_strength(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactRejected("invalid_provenance_strength")
    return value


def segment_and_score(artifact: ValidatedArtifact) -> tuple[ArtifactSegment, ...]:
    """Split a validated artifact at safe line boundaries and score each part alone."""

    if not isinstance(artifact, ValidatedArtifact):
        raise ArtifactRejected("invalid_validated_artifact")
    chunks = _structured_document_chunks(artifact.path, artifact.text)
    if chunks is None:
        chunks = _line_bounded_structural_chunks(artifact.text)
    count = len(chunks)
    output: list[ArtifactSegment] = []
    for index, (text, hard_reasons) in enumerate(chunks):
        assessment = assess_technical_content(text, path=artifact.path, hard_reasons=hard_reasons)
        output.append(
            ArtifactSegment(
                text=text,
                index=index,
                count=count,
                score=assessment.score,
                signals=assessment.signals,
                disposition=assessment.disposition,
                reasons=assessment.reasons,
                raw_sha256=sha256(text.encode("utf-8")).hexdigest(),
                canonical_text_sha256=sha256(
                    canonicalize_text(text).encode("utf-8")
                ).hexdigest(),
                fingerprints=winnowed_five_token_fingerprints(text),
            )
        )
    return tuple(output)


def assess_technical_content(
    text: str, *, path: str | None = None, hard_reasons: tuple[str, ...] = ()
) -> TechnicalAssessment:
    """Assess text alone; a missing path means mixed forum prose, not a file.

    This makes no licensing, provenance or factual-correctness claim. Forum
    language evidence comes only from locally inspected code blocks; metadata
    such as has_code is not an admission signal. Nothing is executed.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if path is None:
        _score, prose_signals = _score_structural_evidence(text, "forum.txt")
        evidence = set(prose_signals)
        for code, suffix in _forum_code_blocks(text):
            _score, code_signals = _score_structural_evidence(code, "block" + suffix)
            evidence.update(code_signals)
        signals = tuple(signal for signal in TECHNICAL_SIGNALS if signal in evidence)
        score = _score_signals(signals)
    else:
        score, signals = _score_structural_evidence(text, path)
    decision, reasons = disposition(score, signals, hard_reasons + _dominance_reasons(text, signals))
    return TechnicalAssessment(score, signals, decision, reasons)


def _forum_code_blocks(text: str) -> Iterator[tuple[str, str]]:
    # Recognize fenced or ordinary indented Python suites, and balanced C/C++
    # function bodies. Do not select a language merely from calls/assignments
    # in prose. Incomplete snippets conservatively remain prose evidence.
    lines = text.splitlines()
    index = 0
    python_start = re.compile(r"^\s*(?:async\s+def|def|class|if|for|while|try|with)\b.*:\s*(?:#.*)?$")
    while index < len(lines):
        line = lines[index]
        fence = re.fullmatch(r"\s*(`{3,}|~{3,})([\w+.-]*)\s*", line)
        if fence:
            end = index + 1
            while end < len(lines) and lines[end].strip() != fence.group(1):
                end += 1
            if end < len(lines):
                code = textwrap.dedent("\n".join(lines[index + 1:end]))
                language = fence.group(2).lower()
                if language in {"", "python", "py"} and _has_valid_python_structure(code):
                    yield code, ".py"
                elif language in {"", "c", "cpp", "c++", "cc", "cxx"} and _is_complete_c_function(code):
                    yield _c_syntax_text(code), ".cpp"
            index = end + 1
            continue
        if python_start.match(line):
            end = index + 1
            while end < len(lines) and (not lines[end].strip() or lines[end][0].isspace()):
                end += 1
            code = textwrap.dedent("\n".join(lines[index:end]))
            if _has_valid_python_structure(code):
                yield code, ".py"
                index = end
                continue
        # Header comments are syntax whitespace. Inspect only this line and
        # its possible opening-brace line, never the surrounding mixed post.
        c_head = _c_syntax_text(line)
        if index + 1 < len(lines):
            next_head = _c_syntax_text(lines[index + 1])
            if next_head.lstrip().startswith("{"):
                c_head += "\n" + next_head
        if _has_definition_or_type(c_head, ".cpp"):
            # Start lexical inspection at a recognized function, never in
            # surrounding prose where apostrophes/quotes are not C literals.
            c_lines = _c_syntax_text("\n".join(lines[index:])).split("\n")
            end = index + 1
            body_started = "{" in c_lines[0]
            balance = c_lines[0].count("{") - c_lines[0].count("}")
            while end < len(lines) and (balance > 0 or not body_started):
                syntax_line = c_lines[end - index]
                body_started = body_started or "{" in syntax_line
                balance += syntax_line.count("{") - syntax_line.count("}")
                end += 1
            code = "\n".join(lines[index:end])
            if _is_complete_c_function(code):
                yield _c_syntax_text(code), ".cpp"
                index = end
                continue
        index += 1


def _is_complete_c_function(text: str) -> bool:
    # Lexical evidence only, not a compiler: require a definition, balanced
    # function body and a terminated statement. Strip strings/comments so
    # braces in them cannot prove a complete body.
    syntax = _c_syntax_text(text)
    balance = 0
    for character in syntax:
        if character == "{":
            balance += 1
        elif character == "}":
            balance -= 1
            if balance < 0:
                return False
    return bool(
        _has_definition_or_type(syntax, ".cpp")
        and balance == 0
        and re.search(r"\b(?:return|throw)\s+[^;{}]+;", syntax)
    )


def _c_syntax_text(text: str) -> str:
    # Preserve line indices and token separation. C literals may cross a
    # physical line only through a backslash escape; comments may span lines.
    return re.sub(
        r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/',
        lambda match: " " + "\n" * match.group().count("\n"),
        text, flags=re.DOTALL,
    )


def disposition(
    score: int, signals: tuple[str, ...], hard_reasons: tuple[str, ...]
) -> tuple[Literal["review_candidate", "quarantined"], tuple[str, ...]]:
    if hard_reasons:
        return "quarantined", hard_reasons
    if score < REVIEW_CANDIDATE_SCORE or len(set(signals)) < 2:
        return "quarantined", ("insufficient_technical_signal",)
    return "review_candidate", ()


def _line_bounded_structural_chunks(text: str) -> list[tuple[str, tuple[str, ...]]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [(text, ())]
    chunks: list[tuple[str, tuple[str, ...]]] = []
    current: list[str] = []
    current_bytes = 0
    previous = ""

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            chunks.append(("".join(current), ()))
            current = []
            current_bytes = 0

    for line_index, line in enumerate(lines):
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > MAX_SEGMENT_BYTES:
            flush()
            chunks.append(("artifact_rejected: overlong_line\n", ("overlong_line",)))
            previous = line
            continue
        if _is_rst_underline(line) and current and previous.strip():
            title = current.pop()
            current_bytes -= len(title.encode("utf-8"))
            flush()
            current.append(title)
            current_bytes = len(title.encode("utf-8"))
        if current and _starts_structural_element(line, previous, line_index):
            flush()
        if current and current_bytes + line_bytes > MAX_SEGMENT_BYTES:
            flush()
        current.append(line)
        current_bytes += line_bytes
        previous = line
    flush()
    return chunks


def _starts_structural_element(line: str, previous: str, line_index: int) -> bool:
    if line_index == 0:
        return False
    if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
        return True
    if re.match(r"^\s*(?:def|class|fn|struct|enum|interface|type)\s+[A-Za-z_]", line):
        return True
    if re.match(
        r"^\s*(?!(?:if|for|while|switch|return|raise|throw)\b)(?:(?:public|private|protected|static|async|const)\s+)*[A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)+\s*\([^;]*\)\s*\{?\s*$",
        line,
    ):
        return True
    if re.match(r"^<(?:[A-Za-z_][\w.-]*)(?:\s|>)", line):
        return True
    if re.match(r"^(?:[A-Za-z_][\w.-]*|['\"][A-Za-z_][\w.-]*['\"])\s*:\s*", line):
        return True
    if re.match(r"^\s*\[[^\]]+\]\s*$", line):
        return True
    if re.match(r"^\s*\[?[A-Za-z_][\w.-]*\]?\s*=", line) and not previous.strip():
        return True
    return False


def _is_rst_underline(line: str) -> bool:
    return bool(re.match(r"^\s*[=\-~^`:#*+]{3,}\s*$", line))


def _structured_document_chunks(
    path: str, text: str
) -> list[tuple[str, tuple[str, ...]]] | None:
    suffix = "." + path.lower().rsplit(".", 1)[-1]
    try:
        if suffix == ".json":
            document = json.loads(text)
            if len(text.encode("utf-8")) <= MAX_SEGMENT_BYTES:
                return [(text, ())]
            return _split_json_document(document)
        if suffix == ".toml":
            tomllib.loads(text)
            return _complete_or_bounded_structured_document(text, "toml")
        if suffix in _XML_EXTENSIONS:
            ElementTree.fromstring(text)
            return _complete_or_bounded_structured_document(text, "xml")
    except (ValueError, ElementTree.ParseError):
        return None
    if suffix in {".yaml", ".yml"}:
        if len(text.encode("utf-8")) <= MAX_SEGMENT_BYTES:
            return [(text, ())]
        return _split_simple_yaml_document(text)
    return None


def _complete_or_bounded_structured_document(
    text: str, kind: str
) -> list[tuple[str, tuple[str, ...]]]:
    if len(text.encode("utf-8")) <= MAX_SEGMENT_BYTES:
        return [(text, ())]
    return [
        (_bounded_structured_representation(kind), ("oversized_configuration_element",))
    ]


def _split_json_document(document: object) -> list[tuple[str, tuple[str, ...]]]:
    if not isinstance(document, dict) or not document:
        return [
            (_bounded_structured_representation("json"), ("oversized_configuration_element",))
        ]
    chunks: list[tuple[str, tuple[str, ...]]] = []
    for key, value in document.items():
        element = json.dumps({key: value}, ensure_ascii=False, separators=(",", ":")) + "\n"
        if len(element.encode("utf-8")) > MAX_SEGMENT_BYTES:
            chunks.append(
                (_bounded_structured_representation("json"), ("oversized_configuration_element",))
            )
        else:
            chunks.append((element, ()))
    return chunks


def _split_simple_yaml_document(text: str) -> list[tuple[str, tuple[str, ...]]]:
    # Without a YAML parser, prove independence only for one-line scalar
    # mappings. Anchors, aliases, flow collections, multiline scalars, tags and
    # directives may depend on another element and must remain one sentinel.
    lines = text.splitlines(keepends=True)
    simple_scalar = re.compile(
        r'''[A-Za-z_][\w.-]*:[ \t]+(?:"[^"\\\x00-\x1f]*"|'(?:[^'\x00-\x1f]|'')*'|[A-Za-z0-9_./+-]+)[ \t]*(?:\#[^\r\n]*)?'''
    )
    if any(
        line.strip() and not line.lstrip().startswith("#")
        and simple_scalar.fullmatch(line.rstrip("\r\n \t")) is None
        for line in lines
    ):
        return [
            (_bounded_structured_representation("yaml"), ("oversized_configuration_element",))
        ]
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^[A-Za-z_][\w.-]*:\s*", line)
    ]
    if not starts:
        return [
            (_bounded_structured_representation("yaml"), ("oversized_configuration_element",))
        ]
    chunks: list[tuple[str, tuple[str, ...]]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        element = canonicalize_text("".join(lines[start:end]))
        if len(element.encode("utf-8")) > MAX_SEGMENT_BYTES:
            chunks.append(
                (_bounded_structured_representation("yaml"), ("oversized_configuration_element",))
            )
        else:
            chunks.append((element, ()))
    return chunks


def _bounded_structured_representation(kind: str) -> str:
    if kind == "json":
        return '{"artifact_rejected":"oversized_configuration_element"}\n'
    if kind == "xml":
        return '<artifact-rejected reason="oversized_configuration_element" />\n'
    if kind == "toml":
        return 'artifact_rejected = "oversized_configuration_element"\n'
    return "artifact_rejected: oversized_configuration_element\n"


def _score_structural_evidence(text: str, path: str) -> tuple[int, tuple[str, ...]]:
    if path.lower().endswith(".py") and not _has_valid_python_structure(text):
        return 0, ()
    suffix = "." + path.lower().rsplit(".", 1)[-1]
    signals: set[str] = set()
    if _has_definition_or_type(text, suffix):
        signals.add("definitions_types")
    calls = re.findall(r"\b[A-Za-z_]\w*\s*\([^()\n]*\)", text)
    non_calls = {"if", "for", "while", "switch", "catch"}
    if suffix in _PROGRAMMING_EXTENSIONS and any(
        call.split("(", 1)[0].strip() not in non_calls for call in calls
    ):
        signals.add("api_calls")
    for signal, pattern in _language_syntax_patterns(suffix):
        if re.search(pattern, text):
            signals.add(signal)
    if re.search(r"(?m)^\s*(?:\$ |>|python |pip |git |npm |cargo )", text):
        signals.add("commands")
    if re.search(r"(?m)^\s*\d+[.)]\s+\S", text):
        signals.add("concrete_procedure")
    if suffix in {".json", ".yaml", ".yml", ".toml", ".xml", ".inf", ".inx"} and len(
        re.findall(r"(?m)^\s*['\"]?[A-Za-z_][\w.-]*['\"]?\s*[:=]\s*\S+", text)
    ) >= 2:
        signals.add("configuration_schema")
    if re.search(r"\b(?:because|validates|checks|decodes)\b.*(?:`[A-Za-z_]\w*`|\b[A-Za-z_]\w*\s*\()", text, re.IGNORECASE):
        signals.add("technical_explanation")
    if re.search(r"\b(?:example|for example)\b.*(?:`[A-Za-z_]\w*`|\b[A-Za-z_]\w*\s*\()", text, re.IGNORECASE):
        signals.add("examples")
    ordered = tuple(signal for signal in TECHNICAL_SIGNALS if signal in signals)
    return _score_signals(ordered), ordered


def _score_signals(signals: tuple[str, ...]) -> int:
    return min(sum(_SIGNAL_WEIGHTS[signal] for signal in set(signals)), 100)


def _language_syntax_patterns(suffix: str) -> tuple[tuple[str, str], ...]:
    """Use only statement forms belonging to the artifact's source language."""
    if suffix not in _PROGRAMMING_EXTENSIONS:
        return ()
    assignment_or_return = r"\breturn\s+\S+|\b[A-Za-z_]\w*\s*=(?!=)\s*[^=\s]+"
    if suffix == ".py":
        control = r"\b(?:if|for|while)\b[^\n:]*:|\byield\s+\S+"
        errors = r"\braise\s+[A-Za-z_]\w*|\b(?:try|except)\b[^\n]*:"
        tests = r"\bassert\s+\S+|\btest_[A-Za-z_]\w*\s*\([^\n)]*\)"
    elif suffix in {".rs", ".go"}:
        if suffix == ".rs":
            control = r"\b(?:if|for|while|match)\b[^\n:{}]*\{"
            errors = r"\b(?:Err\s*\([^()\n]+\)|panic!\s*\([^()\n]*\))"
            tests = r"\b(?:assert|assert_eq|assert_ne)!\s*\([^()\n]+\)|#\[test\]"
        else:
            control = r"\b(?:if|for|switch|select)\b[^\n:{}]*\{"
            errors = r"\b(?:panic|recover)\s*\([^()\n]*\)|\bif\s+err\s*!=\s*nil\s*\{"
            tests = r"\b[a-zA-Z_]\w*\.(?:Fatalf?|Errorf?)\s*\([^()\n]+\)"
    else:
        control = r"\b(?:if|for|while|switch)\s*\([^\n)]*\)\s*\{"
        errors = r"(?!)"
        if suffix not in {".c", ".h"}:
            errors = r"\bthrow\s+(?:new\s+)?[A-Za-z_]\w*|\btry\s*\{|\bcatch\s*\([^\n)]*\)\s*\{"
        tests = r"\bassert\s*\([^()\n]+\)"
        if suffix in {".js", ".ts"}:
            control += r"|\byield\s+\S+"
            tests = r"\b(?:assert|expect|test|it)\s*\([^()\n]+\)"
        elif suffix == ".java":
            tests = r"\bassert\s+[^\n;]+;|\bassert[A-Z]\w*\s*\([^()\n]+\)"
        elif suffix == ".cs":
            tests = r"\b(?:Assert|Debug)\.\w+\s*\([^()\n]+\)"
    return (
        ("control_data_flow", assignment_or_return + "|" + control),
        ("error_handling", errors),
        ("tests", tests),
    )


def _has_definition_or_type(text: str, suffix: str) -> bool:
    if suffix == ".py":
        # The caller has already checked syntax; AST nodes include annotated
        # and async definitions without imposing a second, narrower grammar.
        return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(ast.parse(text)))
    if suffix in {".js", ".ts"}:
        return_annotation = r"(?:\s*:\s*[A-Za-z_$][\w$<>,.\[\]| ?]*)?" if suffix == ".ts" else ""
        return bool(
            re.search(
                rf"^\s*(?:function\s+[A-Za-z_$][\w$]*\s*\([^\n)]*\){return_annotation}\s*\{{|(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=\s*\([^\n)]*\)\s*=>|class\s+[A-Za-z_$][\w$]*)",
                text,
                re.MULTILINE,
            )
        )
    if suffix == ".go":
        return bool(re.search(
            r"^\s*func\s+(?:\([A-Za-z_]\w*\s+\*?[A-Za-z_]\w*\)\s*)?"
            r"[A-Za-z_]\w*\s*\([^\n():]*\)\s*"
            r"(?:\*?(?:\[\])*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?|\([^\n:()]+\))?\s*\{",
            text, re.MULTILINE,
        ))
    if suffix == ".rs":
        return bool(re.search(
            r"^\s*(?:pub(?:\([^\n)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
            r"(?:fn\s+[A-Za-z_]\w*\s*(?:<[^\n>]+>)?\s*\([^\n)]*\)\s*(?:->[^\n{]+)?\{|"
            r"(?:struct|enum|trait)\s+[A-Za-z_]\w*[^\n{;]*\{)",
            text, re.MULTILINE,
        ))
    if suffix in _C_FAMILY_EXTENSIONS:
        return bool(
            re.search(
                r"^\s*(?:(?:public|private|protected|static|const|virtual)\s+)*[A-Za-z_]\w*(?:\s*[*&])?\s+[A-Za-z_]\w*\s*\([^\n;{}]*\)\s*\{",
                text,
                re.MULTILINE,
            )
        )
    return False


def _has_valid_python_structure(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def _dominance_reasons(text: str, signals: tuple[str, ...]) -> tuple[str, ...]:
    marketing_hits = len(
        re.findall(
            r"\b(?:buy now|limited time|incredible|breakthrough|best|effortless|exclusive offer)\b",
            text,
            re.IGNORECASE,
        )
    )
    social_hits = len(
        re.findall(r"\b(?:thanks|awesome work|great job|love this|appreciate it)\b", text, re.IGNORECASE)
    )
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    nonempty_lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    repeated_shape = (
        len(words) >= 60
        and len(set(words)) / len(words) < 0.25
        and len(set(nonempty_lines)) / len(nonempty_lines) < 0.25
        and len(signals) <= 3
    )
    if (marketing_hits >= 8 and marketing_hits > len(signals) * 4) or repeated_shape:
        return ("marketing_dominant",)
    if social_hits >= 8 and social_hits > len(signals) * 4:
        return ("social_dominant",)
    return ()


def winnowed_five_token_fingerprints(text: str) -> frozenset[int]:
    tokens = re.findall(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|->|[^\s]", text.lower())
    if len(tokens) < 5:
        return frozenset()
    hashes = [
        int.from_bytes(sha256("\x1f".join(tokens[index : index + 5]).encode("utf-8")).digest()[:8], "big")
        for index in range(len(tokens) - 4)
    ]
    window_size = 4
    selected: set[int] = set()
    for start in range(max(1, len(hashes) - window_size + 1)):
        window = hashes[start : start + window_size]
        minimum = min(window)
        selected.add(max(index for index, value in enumerate(window, start) if value == minimum))
    return frozenset(hashes[index] for index in selected)


def choose_dedupe_winners(
    segments: Sequence[DedupeRecord], existing: Sequence[DedupeRecord]
) -> tuple[DedupeDecision, ...]:
    """Recompute winners from new and all authenticated, non-tombstoned records.

    ``existing`` must include retained quarantined duplicate losers, not only
    current winners. Similarity is nontransitive: a later stronger winner can
    displace a prior winner while restoring one of its losers. Callers must
    persist and reconstruct those losers with their authenticated score,
    fingerprints and provenance; winner-only history is order-dependent.
    """

    all_records = tuple(segments) + tuple(existing)
    selected_winners: list[DedupeRecord] = []
    decisions: list[DedupeDecision] = []
    for record in sorted(all_records, key=_dedupe_rank):
        winner = next(
            (reference for reference in selected_winners if _records_are_duplicates(record, reference)),
            None,
        )
        if winner is None:
            selected_winners.append(record)
            decisions.append(DedupeDecision(record.provenance_hash, record.provenance_hash, "winner", None))
        else:
            reason = (
                "canonical_text_match"
                if record.segment.canonical_text_sha256 == winner.segment.canonical_text_sha256
                else "near_duplicate"
            )
            decisions.append(DedupeDecision(record.provenance_hash, winner.provenance_hash, "duplicate", reason))
    return tuple(sorted(decisions, key=lambda decision: (decision.provenance_hash, decision.winner_provenance_hash)))


def _records_are_duplicates(left: DedupeRecord, right: DedupeRecord) -> bool:
    if left.segment.canonical_text_sha256 == right.segment.canonical_text_sha256:
        return True
    first, second = left.segment.fingerprints, right.segment.fingerprints
    if not first or not second:
        return False
    intersection = len(first & second)
    union = len(first | second)
    return intersection / union >= 0.92 or intersection / min(len(first), len(second)) >= 0.95


def _dedupe_rank(record: DedupeRecord) -> tuple[int, int, int, str, str]:
    return (
        -record.segment.score,
        -record.provenance_strength,
        len(record.path),
        record.path,
        record.provenance_hash,
    )
