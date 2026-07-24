import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from prama_server.servicer.http import app


client = TestClient(app)


class HttpFrontendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)
        self.frontend_dist = self.temp_path / "dist"
        assets_path = self.frontend_dist / "assets"
        assets_path.mkdir(parents=True)
        (self.frontend_dist / "index.html").write_text(
            "<!doctype html><title>Prama Server</title>",
            encoding="utf-8",
        )
        (assets_path / "app.12345678.js").write_text(
            "console.log('prama');",
            encoding="utf-8",
        )
        self.environment_patch = patch.dict(
            os.environ,
            {"PRAMA_SERVER_WEB_DIST": str(self.frontend_dist)},
        )
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)

    def test_health_endpoint(self) -> None:
        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "name": "Prama ASR Evaluation Service",
                "status": "ok",
            },
        )

    def test_root_serves_frontend_index(self) -> None:
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertIn("Prama Server", response.text)

    def test_frontend_asset_uses_immutable_cache(self) -> None:
        response = client.get("/assets/app.12345678.js")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/javascript")
        )
        self.assertEqual(
            response.headers["cache-control"],
            "public, max-age=31536000, immutable",
        )
        self.assertEqual(response.text, "console.log('prama');")

    def test_frontend_route_falls_back_to_index(self) -> None:
        response = client.get("/evaluations/new")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertIn("Prama Server", response.text)

    def test_missing_static_asset_returns_not_found(self) -> None:
        response = client.get("/assets/missing.js")

        self.assertEqual(response.status_code, 404)

    def test_unknown_api_does_not_fall_back_to_frontend(self) -> None:
        response = client.get("/api/not-found")

        self.assertEqual(response.status_code, 404)
        self.assertTrue(
            response.headers["content-type"].startswith("application/json")
        )
        self.assertEqual(response.json(), {"detail": "API 接口不存在"})

    def test_missing_frontend_dist_keeps_api_available(self) -> None:
        with patch.dict(
            os.environ,
            {"PRAMA_SERVER_WEB_DIST": str(self.temp_path / "missing")},
        ):
            frontend_response = client.get("/")
            health_response = client.get("/api/health")

        self.assertEqual(frontend_response.status_code, 503)
        self.assertEqual(health_response.status_code, 200)

    def test_frontend_does_not_follow_symlink_outside_dist(self) -> None:
        secret_path = self.temp_path / "secret.txt"
        secret_path.write_text("secret", encoding="utf-8")
        (self.frontend_dist / "assets" / "secret.txt").symlink_to(secret_path)

        response = client.get("/assets/secret.txt")

        self.assertEqual(response.status_code, 404)
        self.assertNotEqual(response.text, "secret")
