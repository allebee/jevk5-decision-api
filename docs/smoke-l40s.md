# L40S serving smoke test

2026-09-24, NVIDIA L40S, JevK5 v0.2.0, Hugging Face weight commit `b10f0589014b17597a159ad9318b122280346124`, PyTorch 2.9.1+cu126. The [sample request](../examples/request.json) asks for a ticket team and refund probability in one HTTP call.

| Serving path | Result | First call | Next identical call |
| --- | --- | ---: | ---: |
| Host Python environment with `flash-linear-attention` | HTTP 200; `billing` 0.9937; refund 0.9142; 233 input tokens, 0 output tokens | 155.91 ms | 39.11 ms |
| Built Docker image with reference PyTorch fallback kernels | HTTP 200; `billing` 0.9942; refund 0.9142; 233 input tokens, 0 output tokens | 81.44 ms | 51.52 ms |

The first call is affected by warmup. These are two observations, **not** a latency benchmark or throughput claim. The container's reference kernels are slower and can give slightly different probabilities within normal numerical differences. A `score` request also returned HTTP 200 with a three-level legend and probabilities. The official TypeSafe Python SDK parsed a three-question response from the host service after its `base_url` was pointed at this server.

The container build completed on the same Linux GPU host. The image has not been published to a registry. Startup, concurrent throughput, reliability, and provider billing have not been measured.
