from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import re
import unicodedata

from .models import CapturedDocument


_TOKEN = re.compile(r"\w+", re.UNICODE)
_BOILERPLATE_PHRASES = (
    "access denied",
    "all rights reserved",
    "enable cookies",
    "javascript is disabled",
    "privacy policy",
    "sign in to continue",
    "terms of service",
    "verify you are human",
)
_MAX_64_BIT = (1 << 64) - 1
_SOCIAL_ONLY_TOKENS = frozenset(
    {
        "amazing",
        "appreciate",
        "appreciated",
        "awesome",
        "bro",
        "cheers",
        "confirmed",
        "cool",
        "fixed",
        "for",
        "good",
        "great",
        "helpful",
        "here",
        "it",
        "job",
        "lol",
        "mate",
        "me",
        "much",
        "nice",
        "perfect",
        "really",
        "same",
        "so",
        "solved",
        "thanks",
        "thank",
        "that",
        "this",
        "well",
        "worked",
        "working",
        "works",
        "you",
        "your",
    }
)



@dataclass(frozen=True, slots=True)
class QualityAssessment:
    accepted: bool
    score: float
    reasons: tuple[str, ...]
    normalized_length: int
    alphabetic_ratio: float
    repetition_ratio: float
    boilerplate_hits: tuple[str, ...]
    control_character_count: int


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def simhash64(text: str) -> int:
    tokens = _TOKEN.findall(normalize_text(text))
    if not tokens:
        return 0
    vector = [0] * 64
    for token, weight in Counter(tokens).items():
        hashed = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            vector[bit] += weight if hashed & (1 << bit) else -weight
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(first: int, second: int) -> int:
    for value in (first, second):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("SimHash values must be integers")
        if value < 0 or value > _MAX_64_BIT:
            raise ValueError("SimHash values must be unsigned 64-bit integers")
    return (first ^ second).bit_count()


def _repetition_ratio(tokens: list[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
    repeated = sum(count - 1 for count in Counter(grams).values())
    return repeated / len(grams)


def _is_unsafe_control(character: str) -> bool:
    value = ord(character)
    return (
        (value < 0x20 and value not in (0x09, 0x0A, 0x0D))
        or 0x7F <= value <= 0x9F
        or 0x202A <= value <= 0x202E
        or 0x2066 <= value <= 0x2069
    )


def assess_quality(
    text: str,
    *,
    minimum_length: int = 120,
    minimum_alphabetic_ratio: float = 0.45,
    maximum_repetition_ratio: float = 0.45,
    maximum_boilerplate_hits: int = 1,
) -> QualityAssessment:
    if isinstance(minimum_length, bool) or not isinstance(minimum_length, int):
        raise TypeError("minimum_length must be an integer")
    if minimum_length < 1:
        raise ValueError("minimum_length must be positive")
    if not 0.0 <= minimum_alphabetic_ratio <= 1.0:
        raise ValueError("minimum_alphabetic_ratio must be between zero and one")
    if not 0.0 <= maximum_repetition_ratio <= 1.0:
        raise ValueError("maximum_repetition_ratio must be between zero and one")
    if maximum_boilerplate_hits < 0:
        raise ValueError("maximum_boilerplate_hits cannot be negative")

    normalized = normalize_text(text)
    visible = [character for character in normalized if not character.isspace()]
    alphabetic = sum(character.isalpha() for character in visible)
    alphabetic_ratio = alphabetic / len(visible) if visible else 0.0
    tokens = _TOKEN.findall(normalized)
    repetition_ratio = _repetition_ratio(tokens)
    boilerplate_hits = tuple(
        phrase for phrase in _BOILERPLATE_PHRASES if phrase in normalized
    )
    control_character_count = sum(_is_unsafe_control(character) for character in text)

    reasons: list[str] = []
    if len(normalized) < minimum_length:
        reasons.append("minimum_length")
    if alphabetic_ratio < minimum_alphabetic_ratio:
        reasons.append("alphabetic_ratio")
    if repetition_ratio > maximum_repetition_ratio:
        reasons.append("repetition")
    if len(boilerplate_hits) > maximum_boilerplate_hits:
        reasons.append("boilerplate")
    if control_character_count:
        reasons.append("control_characters")

    length_score = min(1.0, len(normalized) / minimum_length)
    alpha_score = min(
        1.0,
        alphabetic_ratio / minimum_alphabetic_ratio
        if minimum_alphabetic_ratio
        else 1.0,
    )
    repetition_score = max(0.0, 1.0 - repetition_ratio)
    boilerplate_score = max(0.0, 1.0 - 0.2 * len(boilerplate_hits))
    score = round(
        0.30 * length_score
        + 0.30 * alpha_score
        + 0.25 * repetition_score
        + 0.15 * boilerplate_score,
        4,
    )
    if control_character_count:
        score = max(0.0, round(score - 0.25, 4))
    return QualityAssessment(
        accepted=not reasons,
        score=score,
        reasons=tuple(reasons),
        normalized_length=len(normalized),
        alphabetic_ratio=alphabetic_ratio,
        repetition_ratio=repetition_ratio,
        boilerplate_hits=boilerplate_hits,
        control_character_count=control_character_count,
    )


def _is_vbulletin_social_only(text: str) -> bool:
    tokens = _TOKEN.findall(normalize_text(text))
    return bool(tokens) and all(
        token in _SOCIAL_ONLY_TOKENS or token.isdecimal() for token in tokens
    )


def vbulletin_quarantine_reasons(
    document: CapturedDocument,
) -> tuple[str, ...]:
    if not isinstance(document, CapturedDocument):
        raise TypeError("document must be a CapturedDocument")
    if document.metadata.get("source_kind") != "vbulletin_post":
        return ()

    reasons: list[str] = []
    quote_ratio = document.metadata.get("quote_ratio")
    quote_ratio_valid = (
        not isinstance(quote_ratio, bool)
        and isinstance(quote_ratio, Real)
        and 0.0 <= quote_ratio <= 1.0
        and math.isfinite(float(quote_ratio))
    )
    if not quote_ratio_valid:
        reasons.append("vbulletin_quote_ratio_invalid")
    elif quote_ratio > 0.30:
        reasons.append("vbulletin_quote_ratio_exceeds_0.30")

    if type(document.metadata.get("has_code")) is not bool:
        reasons.append("vbulletin_has_code_invalid")
    if _is_vbulletin_social_only(document.raw_text):
        reasons.append("vbulletin_social_only")
    return tuple(reasons)
