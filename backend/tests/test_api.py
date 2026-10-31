from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_solve_structured_response():
    r = client.post("/api/solve", json={"problem": "Solve x^2 - 5x + 6 = 0"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "solved"
    assert body["category"] == "algebra"
    assert body["tool_call"]["tool"] == "solve_equation"
    assert body["answer_text"] == "x = 2, x = 3"
    assert body["verification"]["status"] == "verified"
    assert len(body["steps"]) >= 3
    assert body["visualization"]["data"]
    assert body["session_id"]


def test_validation_errors():
    assert client.post("/api/solve", json={"problem": ""}).status_code == 422
    assert client.post("/api/solve", json={"problem": "x" * 3000}).status_code == 422
    assert client.post("/api/solve", json={}).status_code == 422


def test_unsupported_and_unsafe_inputs():
    for problem in ["write a poem about cats", '__import__("os").system("rm -rf /")']:
        body = client.post("/api/solve", json={"problem": problem}).json()
        assert body["status"] in ("unsupported", "error")
        assert body["answer_text"] is None


def test_followup_compute_and_explain():
    sid = client.post("/api/solve", json={"problem": "derivative of x^3*sin(x)"}).json()["session_id"]
    f = client.post("/api/followup", json={"session_id": sid, "question": "now integrate it"}).json()
    assert f["status"] == "solved"
    assert f["answer_text"] == "x**3*sin(x) + C"
    assert f["verification"]["status"] == "verified"

    g = client.post("/api/followup", json={"session_id": sid, "question": "How was this verified?"}).json()
    assert g["status"] == "answered"
    assert "verification" in g["explanation"].lower()
    assert "scipy.integrate.quad" not in g["explanation"]  # indefinite integral: no quad check


def test_followup_unknown_session():
    assert client.post("/api/followup", json={"session_id": "missing", "question": "why?"}).status_code == 404


def test_examples_all_solve():
    examples = client.get("/api/examples").json()
    for problems in examples.values():
        for p in problems:
            body = client.post("/api/solve", json={"problem": p}).json()
            assert body["status"] == "solved", (p, body["error"])
            assert body["verification"]["status"] in ("verified", "approximate"), p
