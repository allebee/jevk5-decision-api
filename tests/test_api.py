from fastapi.testclient import TestClient

from jevk5_api.app import MODEL_ID, Settings, create_app


class FakeModel:
    def decide(self, state, question):
        assert state == {"ticket": "Charged twice; please refund."}
        if question["type"] == "noul":
            return {"type": "noul", "noul": 0.9, "confidence": 0.9, "input_tokens": 20}
        if question["type"] == "score":
            return {"type": "score", "score": 0.75, "confidence": 0.75,
                    "probabilities": {"0": 0.25, "1": 0.75}, "input_tokens": 30}
        return {"type": "choice", "choice": "billing", "confidence": 0.8,
                "probabilities": {"billing": 0.8, "technical": 0.2}, "input_tokens": 25}


def client():
    return TestClient(create_app(FakeModel(), Settings(api_key="test-secret")))


def payload():
    return {"model": "jevk5", "state": {"ticket": "Charged twice; please refund."},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "Payment", "technical": "Bugs"}},
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "level": {"type": "score", "instructions": "Severity?",
                          "criteria": ["low", "high"]}}}


def test_mixed_decision_contract():
    with client() as api:
        response = api.post("/v1/systemone", headers={"Authorization": "Bearer test-secret"}, json=payload())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == MODEL_ID
        assert body["usage"] == {"input_tokens": 75, "output_tokens": 0}
        assert body["answers"]["team"]["choice"] == "billing"
        assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.9}
        assert body["answers"]["level"]["legend"] == {"0": "low", "1": "high"}
        assert body["answers"]["level"]["score"] == 0.75


def test_authentication_and_health():
    with client() as api:
        assert api.get("/health/live").status_code == 200
        assert api.get("/health/ready").status_code == 200
        assert api.get("/v1/models").json()["data"][0]["id"] == MODEL_ID
        assert api.post("/v1/systemone", json=payload()).status_code == 401
        assert api.post("/v1/systemone", headers={"Authorization": "Bearer wrong"}, json=payload()).status_code == 401


def test_invalid_requests_do_not_run_model():
    with client() as api:
        headers = {"Authorization": "Bearer test-secret"}
        too_many = payload()
        too_many["questions"]["team"]["criteria"] = {f"option_{i}": str(i) for i in range(17)}
        assert api.post("/v1/systemone", headers=headers, json=too_many).status_code == 400
        unknown = payload()
        unknown["model"] = "jev-1.13.0"
        assert api.post("/v1/systemone", headers=headers, json=unknown).status_code == 400
        assert api.post("/v1/systemone", headers=headers, content=b"{").status_code == 400
        assert api.post("/v1/systemone", headers=headers, json={**payload(), "state": "x" * 70000}).status_code == 413
