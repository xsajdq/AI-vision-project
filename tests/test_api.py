from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODELS_DIR = Path(__file__).resolve().parent.parent / "server" / "models_store"
_models_ready = (MODELS_DIR / "vision_autoencoder.pt").exists() and (MODELS_DIR / "audio_autoencoder.pt").exists()

pytestmark = pytest.mark.skipif(
    not _models_ready,
    reason="Run scripts/train_vision.py and scripts/train_audio.py before exercising the API",
)


@pytest.fixture(scope="module")
def client():
    from server.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["vision_model_loaded"] is True
    assert body["audio_model_loaded"] is True


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"QualityScope" in response.content


def test_history_starts_as_list(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
