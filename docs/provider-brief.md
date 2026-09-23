# JevK5 provider brief

## Offer

JevK5 is an open Apache-2.0 4B model for typed decisions over text or JSON state. It returns `choice`, `noul` (probability of true), or `score` plus per-option probabilities. It generates no text. Weights are at [Hugging Face](https://huggingface.co/alibiserikbay/JevK5), training and runtime code at [GitHub](https://github.com/allebee/jevk5), and this repository contains a deployable decision API.

At its September 23 submission, the third-party [JevBench v1.4 board](https://benchmarkheaven.com/jev-models) ranked JevK5 v0.2 #2 of 76 systems, behind TypeSafe's Jev. A separate [36-case public-use-case evaluation](https://github.com/allebee/jevk5/pull/1) found 11/12 expected browser actions, 12/12 document candidate selections, and 10/12 raw tool routes. These cases are small and authored for that evaluation. The tool router's 0.85 confidence gate passed only 1 of 7 actionable requests, so provider users should validate thresholds on their own data.

## Integration facts

| Item | Current service |
| --- | --- |
| Endpoint | `POST /v1/systemone` with bearer authentication |
| Inputs | Text or JSON state; up to 32 typed questions per request |
| Outputs | Jev-style typed answers, per-option probabilities, `input_tokens`, zero `output_tokens` |
| Model | JevK5 v0.2.0; pinned Hugging Face weight revision; 4B bf16 |
| Hardware | One NVIDIA GPU with at least about 9 GB free for weights, plus runtime headroom; tested on L40S |
| Throughput | One process per GPU; inference calls serialized in this reference server; concurrent throughput requires a separate load test |
| Limits | 16 options per choice/score; 128 KiB request; 64 KiB state; long prompts may fall outside captured graph lengths |
| Health | `/health/live`, `/health/ready` |
| Security | Bearer key; ingress must supply TLS, rate limits, and provider-specific controls |

JevK5's `confidence` equals its highest option probability. Jev's confidence can be a separate signal, so a consumer's existing thresholds need fresh calibration.

The [L40S use-case run](https://github.com/allebee/jevk5/blob/codex/jevk5-onboarding-demo/docs/jev-usecases/README.md) recorded roughly 27 ms median for a single local decision after load. This excludes network time and is not a throughput or hosted-latency guarantee. The Docker service still needs a load test and a cost model for any proposed price. No commercial availability or provider approval is implied.

The [serving smoke test](smoke-l40s.md) measured one two-question HTTP request at 39.11 ms warm in the host environment and 51.52 ms warm in the reference Docker image. The image uses slower reference kernels. These are individual observations; provider throughput and tail latency remain unmeasured.

## Provider questions to settle

1. Can the provider expose a native decisions endpoint or custom prediction API? A chat-completion wrapper would change the useful response format.
2. What request/response envelope, authentication, usage accounting, error codes, health probes, and uptime target does that provider require?
3. What deployment region, data retention, traffic volume, and billing agreement apply? Price should follow a measured load test, not a benchmark cost estimate.
4. Will the provider support custom Qwen3.5 weights and JevK5's next-token answer-letter readout? Standard text-generation serving alone will not produce the probability distribution this API promises.

## Outreach draft

Subject: Open Jev-like decision model for your hosted API

Hello,

I built JevK5, an Apache-2.0 4B decision model that returns typed choices and probabilities without generating text. It ranked #2 of 76 on JevBench v1.4. The weights, training code, and a containerized `/v1/systemone` reference server are public:

- Model: https://huggingface.co/alibiserikbay/JevK5
- Code: https://github.com/allebee/jevk5
- Serving: https://github.com/allebee/jevk5-decision-api

I would like to learn whether you can host it as a native decision API, and what model validation, inference, pricing, and reliability requirements you would need from me. I can provide reproducible prompts and GPU measurements. The server currently supports up to 16 options per choice or score question and processes questions sequentially.

Best,
allebee
