from __future__ import annotations

import unittest
import math

from studio.models import CapturedDocument
import studio.quality as quality
from studio.quality import (
    assess_quality,
    hamming_distance,
    normalize_text,
    simhash64,
)


def forum_capture(
    text: str,
    *,
    quote_ratio: object = 0.0,
    has_code: object = False,
) -> CapturedDocument:
    metadata: dict[str, object] = {
        "ingestion": "playwright",
        "source_kind": "vbulletin_post",
        "forum_id": 42,
        "thread_id": 314159,
        "thread_page": 1,
        "post_id": 271828,
        "post_index": 1,
        "has_code": has_code,
        "has_attachment": False,
        "quote_ratio": quote_ratio,
        "original_utf8_bytes": len(text.encode("utf-8")),
        "retained_utf8_bytes": len(text.encode("utf-8")),
        "quote_utf8_bytes": 0,
        "truncated": False,
        "source_text_truncated": False,
    }
    canonical_metadata = dict(metadata)
    nonfinite = type(quote_ratio) is float and not math.isfinite(quote_ratio)
    if nonfinite:
        canonical_metadata["quote_ratio"] = 0.0
    document = CapturedDocument.create(
        blueprint_id="unknowncheats",
        url="https://www.unknowncheats.me/forum/showthread.php?t=314159",
        title="Validated parser discussion",
        raw_text=text,
        metadata=canonical_metadata,
        source_provenance_version=2,
    )
    if nonfinite:
        object.__setattr__(document, "metadata", metadata)
    return document



class QualityTests(unittest.TestCase):
    def test_vbulletin_quote_ratio_uses_strict_greater_than_boundary(self):
        classifier = quality.vbulletin_quarantine_reasons

        at_boundary = forum_capture(
            "The parser validates each field before decoding the retained bytes.",
            quote_ratio=0.30,
        )
        above_boundary = forum_capture(
            "The parser validates each field before decoding the retained bytes.",
            quote_ratio=0.3000001,
        )

        self.assertEqual(classifier(at_boundary), ())
        self.assertEqual(
            classifier(above_boundary),
            ("vbulletin_quote_ratio_exceeds_0.30",),
        )

    def test_vbulletin_metadata_must_be_finite_bounded_and_exactly_typed(self):
        classifier = quality.vbulletin_quarantine_reasons
        technical = (
            "The parser validates the declared record length before decoding fields. "
            "It checks the retained byte count, records the offset, and stops when "
            "the bounded source no longer contains a complete record."
        )
        invalid_quote_ratios = (
            None,
            True,
            "0.20",
            float("nan"),
            float("inf"),
            -0.01,
            10**1000,
            1.01,
        )
        for quote_ratio in invalid_quote_ratios:
            with self.subTest(quote_ratio=quote_ratio):
                self.assertIn(
                    "vbulletin_quote_ratio_invalid",
                    classifier(forum_capture(technical, quote_ratio=quote_ratio)),
                )

        for has_code in (None, 0, 1, "false"):
            with self.subTest(has_code=has_code):
                self.assertIn(
                    "vbulletin_has_code_invalid",
                    classifier(forum_capture(technical, has_code=has_code)),
                )

        self.assertEqual(
            classifier(forum_capture(technical, quote_ratio=0, has_code=True)),
            (),
        )

    def test_vbulletin_missing_metadata_fails_closed_with_stable_reasons(self):
        document = CapturedDocument.create(
            blueprint_id="unknowncheats",
            url="https://www.unknowncheats.me/forum/showthread.php?t=314159",
            title="Validated parser discussion",
            raw_text="The decoder checks every retained byte before reading the next field.",
            metadata={"source_kind": "vbulletin_post"},
            source_provenance_version=2,
        )

        self.assertEqual(
            quality.vbulletin_quarantine_reasons(document),
            (
                "vbulletin_quote_ratio_invalid",
                "vbulletin_has_code_invalid",
            ),
        )

    def test_vbulletin_code_presence_never_erases_quote_reason(self):
        document = forum_capture(
            "Validate the declared size before parsing:\n"
            "if size > len(frame):\n"
            "    raise ValueError('truncated')",
            quote_ratio=0.31,
            has_code=True,
        )

        self.assertEqual(
            quality.vbulletin_quarantine_reasons(document),
            ("vbulletin_quote_ratio_exceeds_0.30",),
        )

    def test_vbulletin_social_only_detection_catches_padded_chatter(self):
        document = forum_capture(
            "Thanks, this works for me. Great job, really appreciate it! " * 6
        )

        self.assertIn(
            "vbulletin_social_only",
            quality.vbulletin_quarantine_reasons(document),
        )

    def test_vbulletin_social_words_do_not_hide_substantive_technical_text(self):
        document = forum_capture(
            "This parser works by comparing the declared packet size with the "
            "captured buffer before decoding the header. It rejects the frame when "
            "the boundary exceeds the retained bytes and records the failed offset."
        )

        self.assertNotIn(
            "vbulletin_social_only",
            quality.vbulletin_quarantine_reasons(document),
        )

    def test_non_vbulletin_capture_keeps_existing_quality_path(self):
        document = CapturedDocument.create(
            blueprint_id="authorized-lab",
            url="https://lab.example/source",
            title="Generic page",
            raw_text="thanks works for me",
            metadata={
                "source_kind": "generic_page",
                "quote_ratio": "malformed",
                "has_code": 1,
            },
        )

        self.assertEqual(quality.vbulletin_quarantine_reasons(document), ())

    def test_normalization_makes_cosmetic_variants_identical(self):
        first = "  CAFÉ\r\npacket\tanalysis  "
        second = "cafe\u0301 packet analysis"

        self.assertEqual(normalize_text(first), "café packet analysis")
        self.assertEqual(normalize_text(first), normalize_text(second))
        self.assertEqual(simhash64(first), simhash64(second))
        self.assertEqual(hamming_distance(simhash64(first), simhash64(second)), 0)

    def test_hamming_distance_rejects_values_outside_64_bits(self):
        with self.assertRaises(ValueError):
            hamming_distance(-1, 0)
        with self.assertRaises(ValueError):
            hamming_distance(0, 1 << 64)

    def test_quality_accepts_substantive_technical_text(self):
        text = (
            "The packet parser validates the Ethernet type before decoding the IPv4 "
            "header. It computes the header length from the low nibble, checks that "
            "the captured frame contains every declared byte, and then presents the "
            "TCP payload as hexadecimal for an operator to review."
        )

        assessment = assess_quality(text)

        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.reasons, ())
        self.assertGreater(assessment.score, 0.75)

    def test_quality_reports_each_low_signal_condition(self):
        cases = {
            "minimum_length": "short note",
            "alphabetic_ratio": "12 34 56 78 90 -- :: !! " * 12,
            "repetition": "repeat this payload marker " * 40,
            "boilerplate": (
                "Access denied. Enable cookies and sign in to continue. "
                "Privacy policy and terms of service. All rights reserved. " * 5
            ),
        }

        for expected_reason, text in cases.items():
            with self.subTest(expected_reason=expected_reason):
                assessment = assess_quality(text)
                self.assertFalse(assessment.accepted)
                self.assertIn(expected_reason, assessment.reasons)

    def test_quality_rejects_unsafe_controls_and_bidi_overrides(self):
        safe_base = (
            "The parser validates each declared record length before decoding fields. "
            "It preserves ordinary line breaks and tabs for readable operator review. "
        )
        for unsafe in ("\x00", "\x1f", "\x85", "\u202e", "\u2066"):
            with self.subTest(codepoint=hex(ord(unsafe))):
                assessment = assess_quality(safe_base + unsafe)
                self.assertFalse(assessment.accepted)
                self.assertIn("control_characters", assessment.reasons)
                self.assertEqual(assessment.control_character_count, 1)

        safe = assess_quality(safe_base + "\n\r\tAdditional reviewed detail.")
        self.assertNotIn("control_characters", safe.reasons)
        self.assertEqual(safe.control_character_count, 0)


if __name__ == "__main__":
    unittest.main()
