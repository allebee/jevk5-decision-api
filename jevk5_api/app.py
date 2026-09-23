"""JevK5 typed-decision API. Serve with one process per GPU."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

MODEL_ID = "alibiserikbay/JevK5"
MODEL_REVISION = "b10f0589014b17597a159ad9318b122280346124"
MODEL_ALIASES = {MODEL_ID, "jevk5", "jev-latest"}
MAX_BODY = 128 * 1024
MAX_STATE = 64 * 1024
MAX_QUESTIONS = 32
MAX_OPTIONS = 16  # The JevK5 v0.2.0 answer-letter head is A through P.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    allow_anonymous: bool = False
    model_revision: str = MODEL_REVISION

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("JEVK5_API_KEY"),
            allow_anonymous=os.getenv("JEVK5_ALLOW_ANONYMOUS") == "1",
            model_revision=os.getenv("JEVK5_MODEL_REVISION", MODEL_REVISION),
        )


def _bad(message: str) -> HTTPException:
    return HTTPException(400, message)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_question(qid: str, question: Any) -> dict:
    if not _nonempty(qid) or len(qid) > 64:
        raise _bad("question ids must be nonempty strings of at most 64 characters")
    if not isinstance(question, dict):
        raise _bad(f"question {qid}: expected an object")
    kind = question.get("type")
    if kind not in {"noul", "choice", "score"}:
        raise _bad(f"question {qid}: type must be noul, choice, or score")
    instructions = question.get("instructions")
    if not (isinstance(instructions, (str, dict, list)) and instructions):
        raise _bad(f"question {qid}: instructions are required")
    if len(json.dumps(instructions, ensure_ascii=False)) > 16 * 1024:
        raise _bad(f"question {qid}: instructions are too long")
    criteria = question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            if not all(_nonempty(item) for item in criteria) or len(set(criteria)) != len(criteria):
                raise _bad(f"question {qid}: choice list must have unique nonempty labels")
            criteria = {label: label for label in criteria}
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise _bad(f"question {qid}: choice needs 2 to {MAX_OPTIONS} options")
        if not all(_nonempty(key) and len(key) <= 64 for key in criteria):
            raise _bad(f"question {qid}: option ids must be nonempty and at most 64 characters")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise _bad(f"question {qid}: score needs 2 to {MAX_OPTIONS} levels")
        if not all(_nonempty(level) for level in criteria):
            raise _bad(f"question {qid}: score levels must be nonempty strings")
    elif criteria is not None:
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise _bad(f"question {qid}: noul criteria must define true and false")
    return {"type": kind, "instructions": instructions, **({"criteria": criteria} if criteria is not None else {})}


def _validate(body: Any) -> tuple[Any, dict[str, dict]]:
    if not isinstance(body, dict):
        raise _bad("request body must be an object")
    requested_model = body.get("model", MODEL_ID)
    if not isinstance(requested_model, str) or requested_model not in MODEL_ALIASES:
        raise _bad("unsupported model; use alibiserikbay/JevK5 or jevk5")
    state = body.get("state")
    if state is None or state == "":
        raise _bad("state is required")
    if len(json.dumps(state, ensure_ascii=False)) > MAX_STATE:
        raise HTTPException(413, "state exceeds 64 KiB")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise _bad(f"questions must contain 1 to {MAX_QUESTIONS} entries")
    return state, {qid: _validate_question(qid, question) for qid, question in questions.items()}


def _answer(model: Any, state: Any, question: dict) -> tuple[dict, int]:
    raw = model.decide(state, question)
    kind = question["type"]
    tokens = raw.get("input_tokens")
    if not isinstance(tokens, int) or tokens < 0:
        raise ValueError("model returned invalid token usage")
    if kind == "noul":
        probability = raw.get("noul")
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("model returned invalid noul probability")
        return {"type": "noul", "noul": probability}, tokens
    probabilities = raw.get("probabilities")
    expected = set(question["criteria"]) if kind == "choice" else {str(i) for i in range(len(question["criteria"]))}
    if not isinstance(probabilities, dict) or set(probabilities) != expected:
        raise ValueError("model returned probabilities for an invalid option set")
    if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("model returned invalid probabilities")
    if abs(sum(probabilities.values()) - 1) > 0.02:
        raise ValueError("model probabilities do not sum to one")
    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("model returned invalid confidence")
    if kind == "choice":
        choice = raw.get("choice")
        if choice not in expected or probabilities[choice] < max(probabilities.values()) - 1e-6:
            raise ValueError("model returned an invalid choice")
        return {"type": kind, "choice": choice, "probabilities": probabilities, "confidence": confidence}, tokens
    score = raw.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= len(expected) - 1:
        raise ValueError("model returned an invalid score")
    return {"type": kind, "score": score, "probabilities": probabilities,
            "legend": {str(i): text for i, text in enumerate(question["criteria"])},
            "confidence": confidence}, tokens


def create_app(model: Any = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.api_key and not settings.allow_anonymous:
            raise RuntimeError("Set JEVK5_API_KEY, or JEVK5_ALLOW_ANONYMOUS=1 for local testing")
        if model is None:
            from huggingface_hub import snapshot_download
            from jevk5 import JevK5

            source = snapshot_download(MODEL_ID, revision=settings.model_revision)
            app.state.model = JevK5(source)
        else:
            app.state.model = model
        yield
        app.state.model = None

    app = FastAPI(title="JevK5 Decision API", version="0.1.0", lifespan=lifespan)

    def authenticate(request: Request) -> None:
        if settings.allow_anonymous:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not settings.api_key or not hmac.compare_digest(supplied, settings.api_key):
            raise HTTPException(401, "valid Bearer token required", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health/live")
    def live():
        return {"ok": True}

    @app.get("/health/ready")
    def ready():
        if getattr(app.state, "model", None) is None:
            return JSONResponse({"ok": False}, status_code=503)
        return {"ok": True, "model": MODEL_ID}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "allebee",
                "capabilities": {"decision_types": ["noul", "choice", "score"],
                                 "max_options_per_question": MAX_OPTIONS,
                                 "max_questions_per_request": MAX_QUESTIONS}}]}

    @app.post("/v1/systemone")
    async def decide(request: Request):
        authenticate(request)
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_BODY:
            raise HTTPException(413, "request exceeds 128 KiB")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY:
                raise HTTPException(413, "request exceeds 128 KiB")
            chunks.append(chunk)
        try:
            body = json.loads(b"".join(chunks))
        except (ValueError, UnicodeDecodeError):
            raise _bad("invalid JSON") from None
        state, questions = _validate(body)
        if getattr(app.state, "model", None) is None:
            raise HTTPException(503, "model is loading")

        def evaluate():
            with lock:  # CUDA graph replay uses mutable static buffers.
                answers, token_count = {}, 0
                for qid, question in questions.items():
                    answers[qid], used = _answer(app.state.model, state, question)
                    token_count += used
                return answers, token_count

        start = time.perf_counter()
        try:
            answers, token_count = await run_in_threadpool(evaluate)
        except Exception as error:
            logger.error("decision failed: %s", type(error).__name__)
            raise HTTPException(500, "inference failed") from None
        return {"id": f"dec-{uuid.uuid4().hex}", "model": MODEL_ID, "answers": answers,
                "usage": {"input_tokens": token_count, "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve JevK5 typed decisions")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run("jevk5_api.app:app", host=args.host, port=args.port, workers=1)
