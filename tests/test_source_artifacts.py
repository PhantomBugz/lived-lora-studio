from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from hashlib import sha256
import tomllib
import unittest
from xml.etree import ElementTree

from studio.source_artifacts import (
    ALLOWED_EXTENSIONS,
    ArtifactSegment,
    ArtifactInput,
    ArtifactRejected,
    DedupeRecord,
    MAX_SEGMENT_BYTES,
    assess_technical_content,
    validate_artifact,
    canonicalize_text,
    choose_dedupe_winners,
    segment_and_score,
)


def artifact(path: str, payload: bytes, *, provenance: object | None = None) -> ArtifactInput:
    return ArtifactInput(
        "github_blob",
        path,
        payload,
        provenance
        if provenance is not None
        else {"owner": "octo", "repository": "demo", "commit_sha": "a" * 40},
        3,
    )


class ForumTechnicalAssessmentTests(unittest.TestCase):
    def test_c_commented_headers_keep_definition_evidence(self):
        for header in (
            "int /* explanation */ mean(int total, int count) {\n",
            "int/* explanation */mean(int total, int count) {\n",
            "int mean(int total, int count) // explanation\n{\n",
        ):
            with self.subTest(header=header):
                text = (
                    "Here's an example that checks `count` because empty collections need a clear result.\n"
                    + header
                    + "    if (count == 0) { return 0; }\n"
                    + "    return total / count;\n}\n"
                    + "I wouldn't divide by zero.\n"
                )
                assessment = assess_technical_content(text)
                self.assertEqual(assessment.disposition, "review_candidate")
                self.assertEqual(assessment.score, 100)
                self.assertIn("definitions_types", assessment.signals)

    def test_c_escaped_newline_keeps_string_braces_nonstructural(self):
        text = (
            "Here's an example that checks `count` because empty collections need a clear result.\n"
            "int mean(int total, int count) {\n"
            "    const char* label = \"line one"
            + chr(92) + "\n"
            + "} stays in the string\";\n"
            + "    if (count == 0) { return 0; }\n"
            + "    return total / count;\n}\n"
            + "I wouldn't divide by zero.\n"
        )
        assessment = assess_technical_content(text)
        self.assertEqual(assessment.disposition, "review_candidate")
        self.assertEqual(assessment.score, 100)
        self.assertIn("definitions_types", assessment.signals)

    def test_c_function_survives_surrounding_contractions_and_quoted_prose(self):
        explanation = "This example checks `count` because empty collections need a clear result.\n"
        code = (
            "int mean(int total, int count) {\n"
            "    if (count == 0) { return 0; }\n"
            "    return total / count;\n"
            "}\n"
        )
        for prefix, suffix in (
            ("Here's the same example.\n", "I wouldn't divide by zero.\n"),
            ('"A reader described the example:\n', 'The procedure is helpful."\n'),
            ("'A reader described the example:\n", "The procedure is helpful.'\n"),
        ):
            with self.subTest(prefix=prefix):
                assessment = assess_technical_content(prefix + explanation + code + suffix)
                self.assertEqual(assessment.disposition, "review_candidate")
                self.assertEqual(assessment.score, 100)
                self.assertIn("definitions_types", assessment.signals)
                self.assertIn("control_data_flow", assessment.signals)

    def test_cpp_contractions_keep_comment_string_and_character_braces_nonstructural(self):
        text = (
            "Here's an example that checks `count` because empty inputs cannot have a mean.\n"
            "double mean(int total, int count)\n"
            "{\n"
            "    const char* label = \"not a boundary }\";\n"
            "    char marker = '}';\n"
            "    /* Not a boundary: } */\n"
            "    if (count == 0) { throw std::invalid_argument(\"empty\"); }\n"
            "    return double(total) / count;\n"
            "}\n"
            "I wouldn't remove the validation.\n"
        )
        assessment = assess_technical_content(text)
        self.assertEqual(assessment.disposition, "review_candidate")
        self.assertEqual(assessment.score, 100)
        self.assertIn("error_handling", assessment.signals)

    def test_forum_line_separators_do_not_crash_assessment(self):
        for separator in ("\r", "\u2028", "\u2029"):
            with self.subTest(separator=repr(separator)):
                text = "People keep arguing about compiler opinions." + separator + "No procedure is supplied."
                try:
                    assessment = assess_technical_content(text)
                except IndexError:
                    self.fail("ordinary line separators must not crash technical admission")
                self.assertEqual(assessment.disposition, "quarantined")
                self.assertEqual(assessment.reasons, ("insufficient_technical_signal",))


class SourceArtifactValidationTests(unittest.TestCase):
    def test_direct_native_credential_literals_reject_without_reflection(self):
        literal = "native-private-fixture"
        fixtures = (
            (".go", 'api_key := "{}"\n'),
            (".go", 'var api_key string = "{}"\n'),
            (".rs", 'let api_key: &str = "{}";\n'),
            (".c", 'const char api_key[] = "{}";\n'),
            (".js", 'const api_key = "{}" // comment\n'),
        )
        for suffix, template in fixtures:
            with self.subTest(suffix=suffix, template=template):
                with self.assertRaises(ArtifactRejected) as caught:
                    validate_artifact(artifact("src/config" + suffix, template.format(literal).encode()))
                self.assertEqual(str(caught.exception), "secret_material")
                for error in _exception_chain(caught.exception):
                    self.assertNotIn(literal, str(error))
                    self.assertNotIn(literal.encode(), getattr(error, "object", b""))

    def test_native_credential_references_placeholders_and_embedded_text_remain_allowed(self):
        fixtures = (
            (".go", 'api_key := configured_api_key\n'),
            (".go", 'var api_key string = os.Getenv("PROVIDER_KEY")\n'),
            (".rs", 'let api_key: &str = load_key();\n'),
            (".c", 'const char *api_key = configured_api_key;\n'),
            (".js", 'const api_key = process.env.PROVIDER_KEY // comment\n'),
            (".go", 'api_key := "REPLACE_ME"\n'),
            (".go", 'var api_key string = "YOUR_API_KEY"\n'),
            (".rs", 'let api_key: &str = "PLACEHOLDER";\n'),
            (".c", 'const char api_key[] = "EXAMPLE";\n'),
            (".js", 'const api_key = "YOUR_API_KEY" // comment\n'),
            (".js", '''const note = 'api_key := "embedded text"';\n'''),
        )
        for suffix, text in fixtures:
            with self.subTest(suffix=suffix, text=text):
                self.assertEqual(validate_artifact(artifact("src/config" + suffix, text.encode())).text, text)

    def test_every_extension_has_a_positive_utf8_fixture(self):
        for extension in sorted(ALLOWED_EXTENSIONS):
            with self.subTest(extension=extension):
                value = validate_artifact(
                    artifact(
                        f"src/example{extension}",
                        b"int parse_value(int value) { return value + 1; }\n",
                    )
                )
                self.assertEqual(value.path, f"src/example{extension}")

    def test_unsafe_names_binary_disguises_and_secrets_fail_with_codes(self):
        cases = {
            "../x.py": (b"print('x')\n", "unsafe_path"),
            "src/x.py.exe": (b"MZ" + b"x" * 200, "unsupported_extension"),
            "src/x.py": (b"\x7fELF" + b"x" * 200, "binary_signature"),
            "src/key.txt": (b"-----BEGIN PRIVATE KEY-----\nsecret", "secret_material"),
        }
        for path, (payload, reason) in cases.items():
            with self.subTest(path=path), self.assertRaisesRegex(ArtifactRejected, reason):
                validate_artifact(artifact(path, payload))

    def test_path_rejections_are_fail_closed(self):
        cases = {
            "/etc/passwd.txt": "unsafe_path",
            "C:/windows/file.txt": "unsafe_path",
            "C:\\windows\\file.txt": "unsafe_path",
            "src\\file.txt": "unsafe_path",
            "./file.txt": "unsafe_path",
            "src/../file.txt": "unsafe_path",
            "src/./file.txt": "unsafe_path",
            "src/file\x00.txt": "unsafe_path",
            "src/file\x1f.txt": "unsafe_path",
            "/".join(["part"] * 65) + ".txt": "unsafe_path",
            "a" * 1021 + ".txt": "unsafe_path",
        }
        for path, reason in cases.items():
            with self.subTest(path=repr(path)), self.assertRaisesRegex(ArtifactRejected, reason):
                validate_artifact(artifact(path, b"safe text\n"))

    def test_generated_dependency_and_unsupported_names_are_rejected(self):
        paths = (
            "src/example.py.exe",
            "src/example.min.js",
            "src/example.js.map",
            "src/__pycache__/example.py",
            "node_modules/pkg/index.js",
            "vendor/pkg/index.go",
            "dist/index.js",
            "build/index.js",
            "package-lock.json",
            "Cargo.lock",
            "poetry.lock",
        )
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(
                ArtifactRejected, "unsupported_extension|generated_or_dependency"
            ):
                validate_artifact(artifact(path, b"const value = 1;\n"))

    def test_binary_signatures_are_rejected_before_decode(self):
        signatures = {
            "pe": b"MZ",
            "elf": b"\x7fELF",
            "macho32": b"\xfe\xed\xfa\xce",
            "macho64": b"\xfe\xed\xfa\xcf",
            "macho32_le": b"\xce\xfa\xed\xfe",
            "macho64_le": b"\xcf\xfa\xed\xfe",
            "zip": b"PK\x03\x04",
            "gzip": b"\x1f\x8b",
            "pdf": b"%PDF-",
            "png": b"\x89PNG\r\n\x1a\n",
            "jpeg": b"\xff\xd8\xff",
            "ole": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
            "sqlite": b"SQLite format 3\x00",
            "lnk": b"L\x00\x00\x00\x01\x14\x02\x00",
        }
        for name, signature in signatures.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ArtifactRejected, "binary_signature"
            ):
                validate_artifact(artifact("src/example.txt", signature + b"x" * 64))

    def test_content_rejections_are_fixed_and_ordered(self):
        cases = {
            "lfs": (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64, "git_lfs_pointer"),
            "nul": (b"valid\x00text", "nul_byte"),
            "invalid_utf8": (b"\xff\xfe", "invalid_utf8"),
            "bom_only": (b"\xef\xbb\xbf", "empty_text"),
            "whitespace_only": (b" \t\r\n", "empty_text"),
            "too_large": (b"x" * (512 * 1024 + 1), "artifact_too_large"),
        }
        for name, (payload, reason) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ArtifactRejected, reason):
                validate_artifact(artifact("src/example.txt", payload))

    def test_secret_shapes_authenticated_urls_and_assignments_are_rejected_without_leaking(self):
        secrets = (
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "glpat-abcdefghijklmnopqrstuvwxyz0123456789",
            "hf_abcdefghijklmnopqrstuvwxyz0123456789",
            "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkZW1vIn0.signaturevalue",
            "https://operator:private-value@example.test/source",
            'api_key = "private-value"',
            'password: "private-value"',
        )
        logger = logging.getLogger("studio.source_artifacts")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            for secret in secrets:
                with self.subTest(secret_kind=secret.split("_", 1)[0]), self.assertRaises(
                    ArtifactRejected
                ) as raised:
                    validate_artifact(artifact("src/example.txt", secret.encode("utf-8")))
                self.assertEqual(raised.exception.reason_code, "secret_material")
                self.assertNotIn("private-value", str(raised.exception))
                self.assertNotIn("private-value", stream.getvalue())
        finally:
            logger.removeHandler(handler)

    def test_inputs_and_provenance_are_immutable_after_validation(self):
        value = validate_artifact(
            artifact(
                "src/example.py",
                b"def parse(value):\n    return value\n",
                provenance={"owner": "octo", "nested": {"labels": ["reviewed"]}},
            )
        )
        with self.assertRaises((AttributeError, TypeError)):
            value.path = "other.py"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            value.provenance["owner"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            value.provenance["nested"]["labels"] = ()  # type: ignore[index]

    def test_invalid_utf8_rejection_chain_never_retains_the_input_bytes(self):
        secret_fixture = b"private-fixture-must-not-leak\xff"
        with self.assertRaises(ArtifactRejected) as raised:
            validate_artifact(artifact("src/example.txt", secret_fixture))
        chain = (raised.exception, raised.exception.__cause__, raised.exception.__context__)
        for error in chain:
            if error is not None:
                self.assertNotIn(secret_fixture, getattr(error, "object", b""))
                self.assertNotIn("private-fixture-must-not-leak", str(error))

    def test_path_validation_precedes_source_kind_and_handles_lone_surrogates(self):
        with self.assertRaisesRegex(ArtifactRejected, "unsafe_path"):
            validate_artifact(ArtifactInput("untrusted", "../example.py", b"pass\n", {}, 1))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ArtifactRejected, "unsafe_path"):
            validate_artifact(ArtifactInput("github_blob", "src/\ud800.py", b"pass\n", {}, 1))

    def test_gif_is_rejected_but_environment_loaded_token_code_is_allowed(self):
        for payload in (b"GIF87a" + b"x" * 32, b"GIF89a" + b"x" * 32):
            with self.subTest(payload=payload[:6]), self.assertRaisesRegex(
                ArtifactRejected, "binary_signature"
            ):
                validate_artifact(artifact("images/example.txt", payload))
        value = validate_artifact(
            artifact(
                "src/config.py",
                b'import os\ntoken = os.environ["PROVIDER_TOKEN"]\n',
            )
        )
        self.assertEqual(value.path, "src/config.py")

    def test_artifact_input_copies_nested_provenance_at_construction(self):
        caller_metadata = {"nested": {"labels": ["reviewed"]}}
        value = ArtifactInput("github_blob", "src/example.py", b"pass\n", caller_metadata, 1)
        caller_metadata["nested"]["labels"].append("changed")
        self.assertEqual(value.provenance["nested"]["labels"], ("reviewed",))

    def test_unquoted_secret_literals_reject_without_leaking_but_benign_forms_pass(self):
        secret_literal = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        with self.assertRaisesRegex(ArtifactRejected, "secret_material") as raised:
            validate_artifact(artifact("src/config.py", f"api_key = {secret_literal}\n".encode()))
        for error in (raised.exception, raised.exception.__cause__, raised.exception.__context__):
            if error is not None:
                self.assertNotIn(secret_literal, str(error))
                self.assertNotIn(secret_literal.encode(), getattr(error, "object", b""))
        benign = (
            b"api_key = configured_api_key\n",
            b'api_key = os.environ["PROVIDER_KEY"]\n',
            b"api_key = load_key()\n",
            b'api_key = "REPLACE_ME"\n',
        )
        for payload in benign:
            with self.subTest(payload=payload):
                self.assertEqual(validate_artifact(artifact("src/config.py", payload)).path, "src/config.py")

    def test_long_identifier_references_are_not_secret_literals(self):
        benign = (
            b"api_key = configured_api_key_from_provider_version2\n",
            b"api_key = provider_api_key_reference\n",
            b'api_key = os.environ["PROVIDER_KEY"]\n',
            b"api_key = resolve_provider_key()\n",
        )
        for payload in benign:
            with self.subTest(payload=payload):
                self.assertEqual(validate_artifact(artifact("src/config.py", payload)).path, "src/config.py")

    def test_bare_yaml_secret_with_underscores_rejects_without_reflection(self):
        literal = "ABCDEF1234567890_GHIJKL0987654321MNOP"
        payload = f"api_key: {literal}\n".encode()
        stream = io.StringIO()
        logger = logging.getLogger("studio.source_artifacts")
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            with self.assertRaises(ArtifactRejected) as raised:
                validate_artifact(artifact("config/provider.yaml", payload))
            self.assertEqual(str(raised.exception), "secret_material")
            self.assertIsNone(raised.exception.__cause__)
            self.assertIsNone(raised.exception.__context__)
            self.assertNotIn(literal, stream.getvalue())
        finally:
            logger.removeHandler(handler)

    def test_structured_credential_values_never_receive_identifier_exemption(self):
        literal = "ABCDEF1234567890_GHIJKL0987654321MNOP"
        fixtures = {
            ".yml": f"api_key: {literal}\n",
            ".toml": f"api_key = {literal}\n",
            ".inf": f"api_key = {literal}\n",
            ".inx": f"api_key = {literal}\n",
            ".txt": f"api_key = {literal}\n",
            ".json": json.dumps({"api_key": literal}),
            ".xml": f"<settings><api_key>{literal}</api_key></settings>\n",
        }
        for suffix, text in fixtures.items():
            with self.subTest(suffix=suffix):
                with self.assertRaises(ArtifactRejected) as raised:
                    validate_artifact(artifact(f"config/provider{suffix}", text.encode()))
                self.assertEqual(str(raised.exception), "secret_material")
                self.assertIsNone(raised.exception.__cause__)
                self.assertIsNone(raised.exception.__context__)

    def test_programming_language_identifier_assignments_remain_accepted(self):
        for suffix in (".py", ".js", ".ts", ".c", ".cpp", ".cs", ".java", ".rs", ".go"):
            with self.subTest(suffix=suffix):
                path = f"src/provider{suffix}"
                payload = b"api_key = configured_api_key_from_provider_version2\n"
                self.assertEqual(validate_artifact(artifact(path, payload)).path, path)

    def test_quoted_data_keys_and_inline_comments_do_not_hide_bare_secrets(self):
        literal = "ABCDEF1234567890_GHIJKL0987654321MNOP"
        for suffix, text in (
            (".yaml", f"api_key: {literal} # provider configuration\n"),
            (".yaml", f'"api_key": {literal}\n'),
            (".toml", f"api_key = {literal} # provider configuration\n"),
            (".toml", f'"api_key" = {literal}\n'),
        ):
            with self.subTest(suffix=suffix, quoted_key=text.startswith('"')):
                with self.assertRaisesRegex(ArtifactRejected, "^secret_material$"):
                    validate_artifact(artifact(f"config/provider{suffix}", text.encode()))

    def test_canonical_hash_removes_one_bom_normalizes_lines_and_preserves_final_line(self):
        value = validate_artifact(
            artifact("src/example.txt", "\ufeffone  \r\ntwo\t\r\n".encode("utf-8"))
        )
        self.assertEqual(canonicalize_text(value.text), "one\ntwo\n")
        self.assertEqual(
            value.canonical_text_sha256,
            sha256(b"one\ntwo\n").hexdigest(),
        )

    def test_credential_declarations_and_flow_values_reject_without_reflection(self):
        literal = "round-five-synthetic-private-value"
        fixtures = (
            (".py", f'api_key = "{literal}"; use(api_key)\n'),
            (".py", f'ready = True; api_key = "{literal}"\n'),
            (".py", f'api_key: str = "{literal}"\n'),
            (".js", f'const api_key = "{literal}";\n'),
            (".ts", f'const api_key: string = "{literal}";\n'),
            (".c", f'const char *api_key = "{literal}";\n'),
            (".cpp", f'std::string api_key = "{literal}";\n'),
            (".java", f'private String password = "{literal}";\n'),
            (".cs", f'string password = "{literal}";\n'),
            (".yaml", f'service: {{api_key: "{literal}", timeout: 5}}\n'),
            (".yml", f'{{"api_key": "{literal}"}}\n'),
            (".toml", f'password = "{literal} with spaces"\n'),
        )
        stream = io.StringIO()
        logger = logging.getLogger("studio.source_artifacts")
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            for index, (suffix, text) in enumerate(fixtures):
                with self.subTest(index=index, suffix=suffix):
                    with self.assertRaises(ArtifactRejected) as raised:
                        validate_artifact(artifact(f"src/provider{suffix}", text.encode()))
                    self.assertEqual(str(raised.exception), "secret_material")
                    for error in _exception_chain(raised.exception):
                        self.assertNotIn(literal, repr(error.args))
                        self.assertNotIn(literal, repr(vars(error)))
                        self.assertNotIn(literal, repr(getattr(error, "object", None)))
                    self.assertIsNone(raised.exception.__cause__)
                    self.assertIsNone(raised.exception.__context__)
                    self.assertNotIn(literal, stream.getvalue())
        finally:
            logger.removeHandler(handler)

    def test_credential_declaration_references_and_placeholders_remain_accepted(self):
        fixtures = (
            (".py", 'ready = True; api_key: str = configured_api_key_from_provider_version2\n'),
            (".py", 'api_key: str = os.environ["PROVIDER_KEY"]; use(api_key)\n'),
            (".py", 'api_key: str = resolve_provider_key()\n'),
            (".js", 'const api_key = process.env.PROVIDER_KEY;\n'),
            (".ts", 'const api_key: string = configured_api_key_from_provider_version2;\n'),
            (".c", 'const char *api_key = getenv("PROVIDER_KEY");\n'),
            (".cpp", 'std::string api_key = resolve_provider_key();\n'),
            (".java", 'private String password = System.getenv("PROVIDER_PASSWORD");\n'),
            (".cs", 'string password = GetConfiguredPassword();\n'),
            (".yaml", 'service: {api_key: "REPLACE_ME", timeout: 5}\n'),
            (".toml", 'password = "YOUR_PASSWORD"\n'),
        )
        for index, (suffix, text) in enumerate(fixtures):
            with self.subTest(index=index, suffix=suffix):
                path = f"src/provider{suffix}"
                self.assertEqual(validate_artifact(artifact(path, text.encode())).text, text)

    def test_surrogate_path_rejection_does_not_retain_input_in_exception_state(self):
        path = "src/private-path-fixture-\ud800.py"
        with self.assertRaises(ArtifactRejected) as raised:
            validate_artifact(artifact(path, b"pass\n"))
        self.assertEqual(str(raised.exception), "unsafe_path")
        for error in _exception_chain(raised.exception):
            self.assertNotIn("private-path-fixture", repr(error.args))
            self.assertNotIn("private-path-fixture", repr(vars(error)))
            self.assertNotIn("private-path-fixture", repr(getattr(error, "object", None)))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)


def validated(text: str, *, path: str = "src/example.py", strength: int = 3):
    return validate_artifact(
        ArtifactInput(
            "github_blob",
            path,
            text.encode("utf-8"),
            {"owner": "octo", "repository": "demo", "commit_sha": "a" * 40},
            strength,
        )
    )


class SourceArtifactScoringTests(unittest.TestCase):
    def test_weak_tail_does_not_inherit_strong_segment_score(self):
        source = validated(
            "def parse(data):\n"
            "    if not data:\n"
            "        raise ValueError('empty')\n"
            "    return decode(data)\n\n"
            "# Thanks, awesome work!\n"
        )
        segments = segment_and_score(source)
        self.assertEqual(segments[0].disposition, "review_candidate")
        self.assertEqual(segments[-1].disposition, "quarantined")
        self.assertIn("insufficient_technical_signal", segments[-1].reasons)

    def test_overlong_line_uses_a_bounded_sentinel_without_partial_original_data(self):
        for text in ("x" * 4_097 + "\n", "\u00e9" * 2_049):
            with self.subTest(byte_count=len(text.encode("utf-8"))):
                self.assertEqual(len(text.encode("utf-8")), 4_098)
                segments = segment_and_score(validated(text))
                self.assertEqual(len(segments), 1)
                self.assertLessEqual(len(segments[0].text.encode("utf-8")), MAX_SEGMENT_BYTES)
                self.assertEqual(segments[0].text, "artifact_rejected: overlong_line\n")
                self.assertEqual(segments[0].disposition, "quarantined")
                self.assertEqual(segments[0].reasons, ("overlong_line",))

    def test_rst_boundaries_split_while_yaml_is_kept_complete_conservatively(self):
        rst = validated(
            "First section\n"
            "-------------\n"
            "This explains `parse`.\n"
            "Second section\n"
            "==============\n"
            "This explains `write`.\n",
            path="notes/review.rst",
        )
        yaml = validated(
            "service:\n"
            "  timeout: 5\n"
            "database:\n"
            "  timeout: 10\n",
            path="config/review.yaml",
        )
        self.assertEqual(
            (len(segment_and_score(rst)), len(segment_and_score(yaml))),
            (2, 1),
        )

    def test_each_segment_has_its_own_raw_and_canonical_hash(self):
        source = validated(
            "def first(value):\n"
            "    return value\n"
            "# Second section\n"
            "def second(value):\n"
            "    return value + 1\n"
        )
        segments = segment_and_score(source)
        first, second = segments[0], segments[-1]
        self.assertNotEqual(first.raw_sha256, second.raw_sha256)
        self.assertNotEqual(first.canonical_text_sha256, second.canonical_text_sha256)

    def test_nontechnical_praise_marketing_speculation_and_keyword_stuffing_are_quarantined(self):
        cases = (
            "Thanks, awesome work. This is amazing and everybody should use it.",
            "The best breakthrough platform: effortless results, incredible value, buy today.",
            "This might probably perhaps solve every possible issue someday.",
            "api parser api parser configuration schema commands tests api parser.",
        )
        for text in cases:
            with self.subTest(text=text[:20]):
                segment = segment_and_score(validated(text, path="notes/review.txt"))[0]
                self.assertEqual(segment.disposition, "quarantined")
                self.assertIn("insufficient_technical_signal", segment.reasons)

    def test_malformed_syntax_and_marketing_dominant_content_are_quarantined(self):
        malformed = validated("def incomplete\ncall(\nraise\nassert\n")
        malformed_segment = segment_and_score(malformed)[0]
        self.assertEqual(malformed_segment.disposition, "quarantined")
        self.assertIn("insufficient_technical_signal", malformed_segment.reasons)

        dominated = validated(
            "def parse(value):\n    return value\n\n"
            + ("Buy now for an incredible breakthrough.\n" * 300),
            path="notes/review.py",
        )
        dominated_segment = segment_and_score(dominated)[0]
        self.assertEqual(dominated_segment.disposition, "quarantined")
        self.assertIn("marketing_dominant", dominated_segment.reasons)

    def test_balanced_malformed_python_and_unlisted_marketing_are_quarantined_but_real_code_passes(self):
        malformed = segment_and_score(validated("def broken():\nreturn fake()\n"))[0]
        self.assertEqual(malformed.disposition, "quarantined")
        marketing = segment_and_score(
            validated(
                "def parse(value):\n    return value\n\n"
                + ("Revolutionary world-class game-changing opportunity.\n" * 300),
                path="notes/review.py",
            )
        )[0]
        self.assertEqual(marketing.disposition, "quarantined")
        self.assertIn("marketing_dominant", marketing.reasons)
        real = segment_and_score(
            validated("def parse(value):\n    if not value:\n        raise ValueError('value')\n    return decode(value)\n")
        )[0]
        self.assertEqual(real.disposition, "review_candidate")

    def test_large_json_and_yaml_segments_stay_bounded_and_structurally_complete(self):
        json_text = '{"first":"' + ("a" * 3_100) + '","second":"' + ("b" * 3_100) + '"}'
        json_text += " " * (6_306 - len(json_text.encode("utf-8")))
        yaml_text = 'first: "' + ("a" * 2_850) + '"\nsecond: "' + ("b" * 2_850) + '"\n'
        yaml_text += " " * (5_796 - len(yaml_text.encode("utf-8")))
        self.assertEqual(len(json_text.encode("utf-8")), 6_306)
        self.assertEqual(len(yaml_text.encode("utf-8")), 5_796)

        json_segments = segment_and_score(validated(json_text, path="config/large.json"))
        yaml_segments = segment_and_score(validated(yaml_text, path="config/large.yaml"))
        for segment in json_segments:
            self.assertLessEqual(len(segment.text.encode("utf-8")), MAX_SEGMENT_BYTES)
            json.loads(segment.text)
        for segment in yaml_segments:
            self.assertLessEqual(len(segment.text.encode("utf-8")), MAX_SEGMENT_BYTES)
            self.assertRegex(segment.text, r"^(?:first|second): \"[a-z]+\"\n?$")

    def test_unsplittable_structured_elements_use_bounded_format_valid_sentinels(self):
        json_text = '{"only":"' + ("a" * 4_990) + '"}'
        json_text += " " * (5_012 - len(json_text.encode("utf-8")))
        xml_text = "<root>" + ("a" * 5_000) + "</root>"
        xml_text += " " * (5_028 - len(xml_text.encode("utf-8")))
        yaml_text = 'only: "' + ("a" * 5_000) + '"\n'
        toml_text = 'only = "' + ("a" * 5_000) + '"\n'
        fixtures = (
            ("config/one.json", json_text, json.loads),
            ("config/one.xml", xml_text, ElementTree.fromstring),
            ("config/one.yaml", yaml_text, None),
            ("config/one.toml", toml_text, tomllib.loads),
        )
        for path, text, parser in fixtures:
            with self.subTest(path=path):
                segments = segment_and_score(validated(text, path=path))
                self.assertTrue(segments)
                for segment in segments:
                    self.assertLessEqual(len(segment.text.encode("utf-8")), MAX_SEGMENT_BYTES)
                    self.assertIn("oversized_configuration_element", segment.reasons)
                    if parser is None:
                        self.assertEqual(
                            segment.text,
                            "artifact_rejected: oversized_configuration_element\n",
                        )
                    else:
                        parser(segment.text)

    def test_python_shaped_nonsense_in_javascript_is_quarantined_while_js_and_c_pass(self):
        nonsense = segment_and_score(
            validated("def broken():\nreturn fake()\n", path="src/broken.js")
        )[0]
        self.assertEqual(nonsense.disposition, "quarantined")
        javascript = segment_and_score(
            validated(
                "function parse(value) {\n"
                "  if (!value) { throw new Error('value'); }\n"
                "  return decode(value);\n"
                "}\n",
                path="src/parser.js",
            )
        )[0]
        c_family = segment_and_score(
            validated(
                "int parse_value(int value) {\n"
                "  if (value < 0) { return -1; }\n"
                "  return decode(value);\n"
                "}\n",
                path="src/parser.c",
            )
        )[0]
        self.assertEqual(javascript.disposition, "review_candidate")
        self.assertEqual(c_family.disposition, "review_candidate")

    def test_python_raise_and_assert_do_not_inflate_javascript_score(self):
        segment = segment_and_score(
            validated("return fake()\nraise ValueError()\nassert result\n", path="src/broken.js")
        )[0]
        self.assertEqual(segment.disposition, "quarantined")
        self.assertNotIn("error_handling", segment.signals)
        self.assertNotIn("tests", segment.signals)

    def test_python_control_error_and_test_forms_are_not_foreign_language_signals(self):
        text = "if ready:\n    process(value)\nraise failure\nassert result\n"
        for suffix in (".js", ".ts", ".c", ".cpp", ".java", ".cs", ".rs", ".go"):
            with self.subTest(suffix=suffix):
                segment = segment_and_score(validated(text, path=f"src/broken{suffix}"))[0]
                self.assertEqual(segment.disposition, "quarantined")
                for signal in ("control_data_flow", "error_handling", "tests"):
                    self.assertNotIn(signal, segment.signals)

    def test_data_and_prose_do_not_receive_programming_syntax_signals(self):
        text = "return fake()\nraise ValueError()\nassert result\n"
        for suffix in (".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".toml", ".xml"):
            with self.subTest(suffix=suffix):
                segment = segment_and_score(validated(text, path=f"notes/example{suffix}"))[0]
                for signal in ("definitions_types", "api_calls", "control_data_flow", "error_handling", "tests"):
                    self.assertNotIn(signal, segment.signals)

    def test_rust_function_definition_contributes_to_positive_scoring(self):
        segment = segment_and_score(
            validated(
                "fn parse(value: i32) -> i32 {\n"
                "    if value < 0 { return 0; }\n"
                "    decode(value)\n"
                "}\n",
                path="src/parser.rs",
            )
        )[0]
        self.assertEqual(segment.disposition, "review_candidate")
        self.assertIn("definitions_types", segment.signals)
        self.assertIn("control_data_flow", segment.signals)
        self.assertIn("api_calls", segment.signals)

    def test_native_language_error_and_test_forms_keep_their_signals(self):
        fixtures = (
            (".js", "throw new Error('value');\nexpect(value);\n"),
            (".cpp", "throw ParseError(value);\nassert(value);\n"),
            (".java", "throw new ParseException(value);\nassert value != null;\n"),
            (".rs", "return Err(parse_error);\nassert!(value > 0);\n"),
            (".go", "panic(value)\nt.Fatal(value)\n"),
        )
        for suffix, text in fixtures:
            with self.subTest(suffix=suffix):
                segment = segment_and_score(validated(text, path=f"src/check{suffix}"))[0]
                self.assertIn("error_handling", segment.signals)
                self.assertIn("tests", segment.signals)

    def test_rust_and_go_do_not_share_foreign_control_keywords(self):
        fixtures = (
            (".go", "while ready {\n  panic(value)\n  t.Fatal(value)\n}\n"),
            (".rs", "switch value {\n  Err(value);\n  assert!(value > 0);\n}\n"),
        )
        for suffix, text in fixtures:
            with self.subTest(suffix=suffix):
                segment = segment_and_score(validated(text, path=f"src/broken{suffix}"))[0]
                self.assertNotIn("control_data_flow", segment.signals)
                self.assertEqual(segment.disposition, "quarantined")

    def test_complete_structured_documents_are_never_split_into_invalid_fragments(self):
        json_source = validated(
            '{\n  "service": {"timeout": 5},\n  "database": {"timeout": 10}\n}\n',
            path="config/review.json",
        )
        toml_source = validated(
            "[service]\ntimeout = 5\n\n[database]\ntimeout = 10\n",
            path="config/review.toml",
        )
        xml_source = validated(
            "<root>\n<service timeout=\"5\" />\n<database timeout=\"10\" />\n</root>\n",
            path="config/review.xml",
        )
        yaml_source = validated(
            "service:\n  timeout: 5\ndatabase:\n  timeout: 10\n",
            path="config/review.yaml",
        )
        for segment in segment_and_score(json_source):
            json.loads(segment.text)
        for segment in segment_and_score(toml_source):
            tomllib.loads(segment.text)
        for segment in segment_and_score(xml_source):
            ElementTree.fromstring(segment.text)
        self.assertEqual(
            tuple(segment.text for segment in segment_and_score(yaml_source)),
            ("service:\n  timeout: 5\ndatabase:\n  timeout: 10\n",),
        )

    def test_xml_project_formats_emit_only_complete_bounded_documents(self):
        text = '<Project>\n<PropertyGroup>\n<Mode>Debug</Mode>\n</PropertyGroup>\n</Project>\n'
        oversized = text.replace("Debug", "x" * 5_000)
        for suffix in (".vcxproj", ".props", ".targets"):
            for payload in (text, oversized):
                with self.subTest(suffix=suffix, oversized=payload is oversized):
                    segments = segment_and_score(validated(payload, path=f"config/project{suffix}"))
                    self.assertEqual(len(segments), 1)
                    for segment in segments:
                        self.assertLessEqual(len(segment.text.encode("utf-8")), MAX_SEGMENT_BYTES)
                        ElementTree.fromstring(segment.text)
                    if payload is oversized:
                        self.assertEqual(segments[0].reasons, ("oversized_configuration_element",))
                        self.assertEqual(segments[0].disposition, "quarantined")
                        self.assertNotIn("xxxx", segments[0].text)
                    else:
                        self.assertEqual(segments[0].text, text)

    def test_oversized_yaml_anchor_dependencies_use_a_complete_bounded_sentinel(self):
        text = 'defaults: &defaults\n  mode: debug\nfirst: *defaults\npadding: "' + "x" * 4_100 + '"\n'
        for suffix in (".yaml", ".yml"):
            with self.subTest(suffix=suffix):
                segments = segment_and_score(validated(text, path=f"config/anchors{suffix}"))
                self.assertEqual(len(segments), 1)
                self.assertLessEqual(len(segments[0].text.encode("utf-8")), MAX_SEGMENT_BYTES)
                self.assertEqual(segments[0].text, "artifact_rejected: oversized_configuration_element\n")
                self.assertEqual(segments[0].reasons, ("oversized_configuration_element",))
                self.assertEqual(segments[0].disposition, "quarantined")

    def test_oversized_yaml_ambiguous_multiline_forms_are_not_partially_split(self):
        fixtures = (
            'mapping: {\nfirst: 1,\nsecond: 2\n}\n',
            'message: "first line\nsecond: still the same scalar"\n',
        )
        for index, prefix in enumerate(fixtures):
            with self.subTest(index=index):
                text = prefix + 'padding: "' + "x" * 4_100 + '"\n'
                segments = segment_and_score(validated(text, path="config/multiline.yaml"))
                self.assertEqual(len(segments), 1)
                self.assertEqual(segments[0].text, "artifact_rejected: oversized_configuration_element\n")
                self.assertEqual(segments[0].reasons, ("oversized_configuration_element",))

    def test_native_go_annotated_python_and_typed_typescript_definitions_score(self):
        fixtures = (
            (".go", "func parse(value int) int {\n    if value < 0 { return 0 }\n    return decode(value)\n}\n"),
            (".py", "def parse(value: int) -> int:\n    if value < 0:\n        return 0\n    return decode(value)\n"),
            (".ts", "function parse(value: number): number {\n    if (value < 0) { return 0; }\n    return decode(value);\n}\n"),
        )
        for suffix, text in fixtures:
            with self.subTest(suffix=suffix):
                segments = segment_and_score(validated(text, path=f"src/parser{suffix}"))
                self.assertEqual(len(segments), 1)
                self.assertIn("definitions_types", segments[0].signals)
                self.assertIn("api_calls", segments[0].signals)
                self.assertIn("control_data_flow", segments[0].signals)
                self.assertEqual(segments[0].disposition, "review_candidate")
            for foreign_suffix in (".js", ".c", ".rs", ".go", ".py"):
                if foreign_suffix == suffix:
                    continue
                with self.subTest(native=suffix, foreign=foreign_suffix):
                    segments = segment_and_score(validated(text, path=f"src/foreign{foreign_suffix}"))
                    for segment in segments:
                        self.assertNotIn("definitions_types", segment.signals)
                        self.assertEqual(segment.disposition, "quarantined")

    def test_structural_scoring_records_behavioral_evidence_not_terms_alone(self):
        source = validated(
            "def load_config(path):\n"
            "    data = json.loads(path.read_text())\n"
            "    if 'timeout' not in data:\n"
            "        raise ValueError('timeout is required')\n"
            "    assert data['timeout'] > 0\n"
            "    return data\n"
        )
        segments = segment_and_score(source)
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.disposition, "review_candidate")
        self.assertGreaterEqual(segment.score, 55)
        self.assertIn("definitions_types", segment.signals)
        self.assertIn("api_calls", segment.signals)
        self.assertIn("control_data_flow", segment.signals)
        self.assertIn("error_handling", segment.signals)
        self.assertIn("tests", segment.signals)

    def test_dedupe_winner_is_independent_of_collection_order(self):
        weak_long_path = DedupeRecord(
            _segment("weak", 56, frozenset({10, 11, 12})),
            "very/long/path/to/a/reviewed/artifact.py",
            "a" * 64,
            1,
        )
        strong_short_path = DedupeRecord(
            _segment("strong", 80, frozenset({10, 11, 12})),
            "a.py",
            "b" * 64,
            3,
        )
        first = choose_dedupe_winners((weak_long_path, strong_short_path), ())
        second = choose_dedupe_winners((strong_short_path, weak_long_path), ())
        self.assertEqual(_winner(first).provenance_hash, _winner(second).provenance_hash)
        self.assertEqual(_winner(first).provenance_hash, "b" * 64)

    def test_existing_authenticated_record_uses_the_same_canonical_comparison(self):
        current = DedupeRecord(
            _segment("current", 70, frozenset()), "incoming.py", "c" * 64, 2
        )
        active = DedupeRecord(
            _segment("active", 75, frozenset()), "saved.py", "d" * 64, 3
        )
        decision = choose_dedupe_winners((current,), (active,))[0]
        self.assertEqual(decision.disposition, "duplicate")
        self.assertEqual(decision.winner_provenance_hash, "d" * 64)
        self.assertEqual(decision.reason, "canonical_text_match")

    def test_stronger_new_winner_emits_a_decision_for_an_existing_loser(self):
        new = DedupeRecord(_segment("new", 80, frozenset()), "new.py", "e" * 64, 3)
        existing = DedupeRecord(
            _segment("existing", 70, frozenset()), "old.py", "f" * 64, 2
        )
        decisions = choose_dedupe_winners((new,), (existing,))
        existing_decision = next(value for value in decisions if value.provenance_hash == "f" * 64)
        self.assertEqual(existing_decision.disposition, "duplicate")
        self.assertEqual(existing_decision.winner_provenance_hash, "e" * 64)

    def test_nontransitive_similarity_keeps_unmatched_high_rank_records_as_winners(self):
        first = DedupeRecord(_segment("first", 90, frozenset(range(0, 100)), "1" * 64), "a.py", "a" * 64, 3)
        bridge = DedupeRecord(_segment("bridge", 70, frozenset(range(0, 108)), "2" * 64), "b.py", "b" * 64, 3)
        third = DedupeRecord(_segment("third", 80, frozenset(range(8, 108)), "3" * 64), "c.py", "c" * 64, 3)
        decisions = choose_dedupe_winners((first, bridge, third), ())
        self.assertEqual(
            {value.provenance_hash for value in decisions if value.disposition == "winner"},
            {"a" * 64, "c" * 64},
        )
        bridge_decision = next(value for value in decisions if value.provenance_hash == "b" * 64)
        self.assertEqual(bridge_decision.winner_provenance_hash, "a" * 64)

    def test_near_duplicate_fingerprints_and_tie_breakers_are_deterministic(self):
        first = DedupeRecord(
            _segment("one", 70, frozenset(range(100, 200))), "z.py", "e" * 64, 3
        )
        second = DedupeRecord(
            _segment("two", 70, frozenset(range(100, 199))), "a.py", "f" * 64, 3
        )
        one_transaction = choose_dedupe_winners((first, second), ())
        another_transaction = choose_dedupe_winners((second, first), ())
        self.assertEqual(_winner(one_transaction).provenance_hash, "f" * 64)
        self.assertEqual(_winner(one_transaction).provenance_hash, _winner(another_transaction).provenance_hash)

    def test_cross_run_dedupe_reconsiders_authenticated_retained_duplicate_losers(self):
        first = DedupeRecord(_segment("A", 70, frozenset(range(100)), "1" * 64), "a.py", "a" * 64, 3)
        bridge = DedupeRecord(_segment("B", 80, frozenset(range(108)), "2" * 64), "b.py", "b" * 64, 3)
        third = DedupeRecord(_segment("C", 90, frozenset(range(8, 108)), "3" * 64), "c.py", "c" * 64, 3)
        all_at_once = choose_dedupe_winners((first, bridge, third), ())
        expected_winners = {"a" * 64, "c" * 64}
        self.assertEqual({value.provenance_hash for value in all_at_once if value.disposition == "winner"}, expected_winners)
        for initial in ((first, bridge), (bridge, first)):
            initial_decisions = choose_dedupe_winners(initial, ())
            self.assertEqual(_winner(initial_decisions).provenance_hash, "b" * 64)
            retained = tuple(
                replace(record, segment=replace(record.segment, disposition="quarantined", reasons=("near_duplicate",)))
                if record is first else record
                for record in initial
            )
            self.assertEqual(choose_dedupe_winners((third,), retained), all_at_once)
        winner_only = choose_dedupe_winners((third,), (bridge,))
        self.assertEqual({value.provenance_hash for value in winner_only if value.disposition == "winner"}, {"c" * 64})
        self.assertNotEqual(winner_only, all_at_once)


def _segment(
    label: str,
    score: int,
    fingerprints: frozenset[int],
    canonical_text_sha256: str = "2" * 64,
) -> ArtifactSegment:
    return ArtifactSegment(
        text=label,
        index=0,
        count=1,
        score=score,
        signals=("definitions_types", "api_calls"),
        disposition="review_candidate",
        reasons=(),
        raw_sha256="1" * 64,
        canonical_text_sha256=canonical_text_sha256,
        fingerprints=fingerprints,
    )


def _winner(decisions):
    return next(value for value in decisions if value.disposition == "winner")


def _exception_chain(error):
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend((current.__cause__, current.__context__))
