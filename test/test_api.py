from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

def test_health():
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_predict_valid():
    payload = {
        "title": "Login crash",
        "description": "Users get 500 after OAuth callback"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "priority" in body
    assert "confidence" in body
    assert "top_terms" in body

def test_predict_empty():
    payload = {"title": "", "description": ""}
    res = client.post("/predict", json=payload)
    assert res.status_code == 400

if __name__ == "__main__":
    test_health()
    test_predict_valid()
    test_predict_empty()
