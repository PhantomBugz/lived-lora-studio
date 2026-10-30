from __future__ import annotations

import json
import unittest

from studio.blueprints import (
    BUILTIN_BLUEPRINT_IDS,
    BUILTIN_BLUEPRINTS,
    BlueprintValidationError,
    SourceBlueprint,
    UnknownCheatsBlueprint,
    UrlScopeError,
    generic_blueprint,
    normalize_https_url,
    require_registered_blueprint_url,
)
from studio.vbulletin import VbulletinCrawlLimits


def valid_blueprint_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "blueprint_id": "example-docs",
        "name": "Example documentation",
        "allowed_hosts": ["example.com"],
        "login_hosts": ["example.com", "auth.example.net"],
        "content_selector": "main article, main .documentation",
        "title_selector": "main h1",
        "link_selector": "main a[href]",
        "max_pages": 12,
        "crawl_delay_seconds": 0.5,
    }


class SourceBlueprintTests(unittest.TestCase):
    def test_strict_json_loader_accepts_the_complete_schema(self):
        blueprint = SourceBlueprint.from_json(json.dumps(valid_blueprint_mapping()))

        self.assertEqual(blueprint.blueprint_id, "example-docs")
        self.assertEqual(blueprint.allowed_hosts, ("example.com",))
        self.assertEqual(
            blueprint.login_hosts,
            ("example.com", "auth.example.net"),
        )
        self.assertEqual(blueprint.max_pages, 12)
        self.assertEqual(blueprint.crawl_delay_seconds, 0.5)

    def test_strict_json_loader_rejects_missing_unknown_and_wrong_typed_fields(self):
        missing = valid_blueprint_mapping()
        del missing["content_selector"]
        extra = valid_blueprint_mapping() | {"script": "click-login"}
        bool_page_limit = valid_blueprint_mapping() | {"max_pages": True}
        wrong_hosts = valid_blueprint_mapping() | {"allowed_hosts": "example.com"}
        wrong_login_hosts = valid_blueprint_mapping() | {
            "login_hosts": "auth.example.net"
        }

        for value in (
            missing,
            extra,
            bool_page_limit,
            wrong_hosts,
            wrong_login_hosts,
        ):
            with self.subTest(value=value), self.assertRaises(BlueprintValidationError):
                SourceBlueprint.from_json(json.dumps(value))

    def test_blueprint_rejects_invalid_hosts_selectors_and_page_bounds(self):
        invalid_values = (
            {"allowed_hosts": ["https://example.com"]},
            {"allowed_hosts": ["example.com.evil/"]},
            {"login_hosts": ["auth.example.net"]},
            {"login_hosts": ["example.com", "https://auth.example.net"]},
            {"content_selector": "xpath=//article"},
            {"content_selector": "article { display: none; }"},
            {"title_selector": "main["},
            {"max_pages": 0},
            {"max_pages": 101},
            {"crawl_delay_seconds": -0.1},
        )

        for change in invalid_values:
            value = valid_blueprint_mapping() | change
            with self.subTest(change=change), self.assertRaises(BlueprintValidationError):
                SourceBlueprint.from_mapping(value)

    def test_allowed_urls_require_https_and_an_exact_listed_host(self):
        blueprint = SourceBlueprint.from_mapping(
            valid_blueprint_mapping()
            | {
                "allowed_hosts": ["example.com", "docs.example.com"],
                "login_hosts": [
                    "example.com",
                    "docs.example.com",
                    "auth.example.net",
                ],
            }
        )

        self.assertEqual(
            blueprint.require_allowed_url("HTTPS://EXAMPLE.COM:443/guide#part"),
            "https://example.com/guide",
        )
        self.assertEqual(
            blueprint.require_allowed_url("/next", base_url="https://example.com/guide/start"),
            "https://example.com/next",
        )
        for url in (
            "http://example.com/guide",
            "https://api.example.com/guide",
            "https://example.com.evil/guide",
            "https://user:password@example.com/guide",
            "https://example.com:444/guide",
        ):
            with self.subTest(url=url), self.assertRaises(UrlScopeError):
                blueprint.require_allowed_url(url)

    def test_url_normalization_removes_fragments_and_dot_segments(self):
        self.assertEqual(
            normalize_https_url("../reference/./index.html#section", base_url="https://EXAMPLE.com/a/b/"),
            "https://example.com/a/reference/index.html",
        )

    def test_registered_target_url_requires_an_exact_existing_blueprint(self):
        self.assertEqual(
            require_registered_blueprint_url(
                "tryhackme",
                "https://tryhackme.com/room/authorized-packet-lab",
            ),
            "https://tryhackme.com/room/authorized-packet-lab",
        )
        for blueprint_id, url in (
            ("missing", "https://tryhackme.com/room/authorized-packet-lab"),
            ("tryhackme", "https://example.com/room/not-authorized"),
        ):
            with self.subTest(blueprint_id=blueprint_id, url=url), self.assertRaises(
                (BlueprintValidationError, UrlScopeError)
            ):
                require_registered_blueprint_url(blueprint_id, url)

        custom = generic_blueprint(
            host="authorized-lab.example",
            content_selector="main article",
        )
        self.assertEqual(
            require_registered_blueprint_url(
                custom.blueprint_id,
                "https://authorized-lab.example/packet-analysis",
                blueprints={custom.blueprint_id: custom},
            ),
            "https://authorized-lab.example/packet-analysis",
        )

    def test_url_validation_rejects_encoded_controls_and_bidi_text(self):
        for url in (
            "https://tryhackme.com/room/%0aheader",
            "https://tryhackme.com/room/%7fdelete",
            "https://tryhackme.com/room/bad\u202etext",
            "https://tryhackme.com/room/bad\u061ctext",
        ):
            with self.subTest(url=repr(url)), self.assertRaises(UrlScopeError):
                require_registered_blueprint_url("tryhackme", url)

    def test_zero_delay_is_valid_for_an_injected_or_offline_session(self):
        blueprint = SourceBlueprint.from_mapping(
            valid_blueprint_mapping() | {"crawl_delay_seconds": 0.0}
        )

        self.assertEqual(blueprint.crawl_delay_seconds, 0.0)

    def test_builtin_and_generic_blueprints_cover_declared_sources(self):
        self.assertEqual(
            BUILTIN_BLUEPRINT_IDS,
            ("unknowncheats", "hackerone", "tryhackme", "generic"),
        )
        self.assertEqual(set(BUILTIN_BLUEPRINTS), {"unknowncheats", "hackerone", "tryhackme"})
        self.assertTrue(
            BUILTIN_BLUEPRINTS["unknowncheats"].allows_url(
                "https://www.unknowncheats.me/forum/thread.html"
            )
        )
        self.assertTrue(
            BUILTIN_BLUEPRINTS["hackerone"].allows_url("https://hackerone.com/reports/1")
        )
        self.assertTrue(
            BUILTIN_BLUEPRINTS["tryhackme"].allows_url("https://tryhackme.com/room/example")
        )

        generic = generic_blueprint(
            host="research.example.org",
            login_hosts=("login.example-id.net",),
            content_selector="main .technical-content",
            title_selector="main h1",
            max_pages=3,
        )
        self.assertEqual(generic.blueprint_id, "generic-research-example-org")
        self.assertTrue(generic.allows_url("https://research.example.org/writeup"))
        self.assertFalse(generic.allows_url("https://cdn.research.example.org/writeup"))
        self.assertEqual(
            generic.login_hosts,
            ("research.example.org", "login.example-id.net"),
        )

    def test_unknowncheats_builtin_uses_the_typed_bounded_forum_strategy(self):
        blueprint = BUILTIN_BLUEPRINTS["unknowncheats"]

        self.assertIsInstance(blueprint, UnknownCheatsBlueprint)
        self.assertEqual(blueprint.strategy, "vbulletin")
        self.assertEqual(
            blueprint.vbulletin_limits,
            VbulletinCrawlLimits(25, 100, 25, 50, 1_000),
        )
        self.assertEqual(
            blueprint.endpoint_paths,
            ("/forum/forumdisplay.php", "/forum/showthread.php"),
        )
        self.assertEqual(
            blueprint.content_selector,
            "#posts [id^='post_message_'], #posts .post_message",
        )
        self.assertEqual(blueprint.to_mapping()["schema_version"], 1)
        self.assertNotIn("strategy", blueprint.to_mapping())
        self.assertNotIn("vbulletin_limits", blueprint.to_mapping())

    def test_unknowncheats_bare_host_is_canonicalized_to_www(self):
        blueprint = BUILTIN_BLUEPRINTS["unknowncheats"]

        self.assertEqual(
            blueprint.require_allowed_url(
                "https://unknowncheats.me/forum/forumdisplay.php?f=185"
            ),
            "https://www.unknowncheats.me/forum/forumdisplay.php?f=185",
        )
        self.assertEqual(
            blueprint.require_allowed_url(
                "?f=185&page=2",
                base_url="https://unknowncheats.me/forum/forumdisplay.php?f=185",
            ),
            "https://www.unknowncheats.me/forum/forumdisplay.php?f=185&page=2",
        )
        canonical = "https://www.unknowncheats.me/forum/showthread.php?t=77&page=2"
        self.assertEqual(blueprint.require_allowed_url(canonical), canonical)

    def test_unknowncheats_blueprint_rejects_operator_defined_identity_and_host(self):
        with self.assertRaises(BlueprintValidationError):
            UnknownCheatsBlueprint(
                schema_version=1,
                blueprint_id="operator-forum",
                name="Operator forum",
                allowed_hosts=("example.org",),
                login_hosts=("example.org",),
                content_selector="main article",
                title_selector="main h1",
                link_selector="a[href]",
                max_pages=1,
                crawl_delay_seconds=0.0,
            )


if __name__ == "__main__":
    unittest.main()
