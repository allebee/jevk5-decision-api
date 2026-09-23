# JevK5 Decision API

A separately deployable HTTP service for [JevK5](https://github.com/allebee/jevk5), the open 4B typed-decision model. It answers `noul`, `choice`, and `score` questions with probabilities and **zero generated text tokens**. The service is intended for inference providers evaluating whether to host the model and for developers who want their own endpoint.

The request and answer shapes follow TypeSafe's [System One quickstart](https://docs.typesafe.ai/introduction/quickstart). This is an independent JevK5 service, not TypeSafe or OpenRouter infrastructure. It implements the documented decision fields; it does not claim full SDK or provider certification.

## Run on one NVIDIA GPU

JevK5 v0.2.0 uses about 9 GB of GPU memory in bf16. Docker with the NVIDIA Container Toolkit is required:

```sh
docker build -t jevk5-decision-api .
export JEVK5_API_KEY='replace-with-a-long-random-secret'
docker run --rm --gpus all -p 127.0.0.1:8090:8090 \
  -e JEVK5_API_KEY -v jevk5-model-cache:/models/huggingface \
  jevk5-decision-api
```

The image pins JevK5's `v0.2.0` package. The service pins the public [Hugging Face weights](https://huggingface.co/alibiserikbay/JevK5) to commit `b10f0589014b17597a159ad9318b122280346124` by default. First startup downloads the weights. Set `JEVK5_MODEL_REVISION` to change that pin. Keep one server worker per GPU: the CUDA graph runtime uses shared buffers and the API serializes inference calls.

```sh
curl -s http://127.0.0.1:8090/health/ready
curl -s http://127.0.0.1:8090/v1/systemone \
  -H "Authorization: Bearer $JEVK5_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "alibiserikbay/JevK5",
    "state": {"ticket": "I was charged twice this month. Please refund the duplicate."},
    "questions": {
      "team": {"type": "choice", "instructions": "Which team should handle the ticket?",
        "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"}},
      "refund": {"type": "noul", "instructions": "Does the customer request a refund?"}
    }
  }'
```

The response contains `model`, `answers`, `usage.input_tokens`, `usage.output_tokens: 0`, a request ID, and elapsed server time. `choice` answers contain the selected option, all option probabilities, and confidence. `score` answers contain an expected score, legend, probabilities, and confidence. `noul` returns the probability of true. One request may contain multiple questions, processed sequentially. `model` accepts `alibiserikbay/JevK5`, `jevk5`, or `jev-latest` as input; the response always identifies JevK5.

JevK5 currently reports the highest option probability as `confidence`; Jev's separate confidence field can behave differently. Recalibrate any application threshold before replacing Jev.

The official TypeSafe Python SDK was smoke-tested against this endpoint by changing its base URL:

```python
import os
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(
    api_key=os.environ["JEVK5_API_KEY"],
    base_url="http://127.0.0.1:8090",
)
result = client.system_one(
    model="jev-latest",  # Accepted alias; response.model identifies JevK5.
    state="Charged twice; please refund the duplicate.",
    questions={"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}},
)
print(result.answers["refund"].noul)
```

## Deploy and operate

- `/health/live` checks the process; `/health/ready` responds only after weights load and graph capture finishes.
- `/v1/models` lists the model and its decision capabilities. It is descriptive metadata, not a claim of OpenRouter provider compatibility.
- `/v1/systemone` requires a bearer key unless `JEVK5_ALLOW_ANONYMOUS=1` is set for local testing. Use TLS and rate limits at your ingress for public traffic. The service does not store prompts or answers; your platform's access logs, proxy, and retention settings still matter.
- Requests are limited to 128 KiB, state to 64 KiB, 32 questions, and 16 options per `choice` or `score`. The model may be slower for long inputs; the fast captured graph lengths stop at 4,096 tokens.
- The included [provider brief](docs/provider-brief.md) states the current evidence and gaps. See [JevK5's use-case report](https://github.com/allebee/jevk5/pull/1) for application tests.
- The [L40S smoke record](docs/smoke-l40s.md) includes real HTTP results from the host service and built container. The container uses slower reference kernels because it does not install `flash-linear-attention`.

For a direct Python installation on a GPU host:

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install '.[gpu]'
JEVK5_API_KEY='replace-with-a-long-random-secret' jevk5-api --host 127.0.0.1 --port 8090
```

Run API contract tests without a GPU using `python -m pip install '.[test]' && pytest -q`.

## Provider status

This repository is a working decision endpoint and packaging starting point. **It has not been accepted by OpenRouter, Codiv, Replicate, or any other provider.** OpenRouter currently exposes Jev through a [Decisions API](https://openrouter.ai/blog/insights/what-is-jev/), while its [standard provider application](https://openrouter.ai/providers/apply) specifies chat-completion requirements. Decision-model onboarding therefore needs a direct discussion with them. See [the outreach draft](docs/provider-brief.md#outreach-draft).

Apache-2.0. The JevK5 model weights and training code have their own [repository](https://github.com/allebee/jevk5).
