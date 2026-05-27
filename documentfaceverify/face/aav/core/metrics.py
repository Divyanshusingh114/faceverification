"""
metrics.py
----------
Prometheus counters / histograms specific to the verification flow.

HTTP-level metrics (requests, latency by route) come from
prometheus-fastapi-instrumentator wired in app.py. The metrics here are
domain-specific so we can alert on, e.g., a sudden surge of REJECTED
decisions or a spike in `face_not_consistently_visible` liveness failures.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

decisions_total = Counter(
    "aav_decisions_total",
    "Verification decisions, by tier",
    labelnames=("tier",),
)

liveness_fail_total = Counter(
    "aav_liveness_failures_total",
    "Liveness check failures, by reason code",
    labelnames=("reason",),
)

upload_rejections_total = Counter(
    "aav_upload_rejections_total",
    "Card-upload rejections, by reason code",
    labelnames=("reason",),
)

verify_latency_seconds = Histogram(
    "aav_verify_seconds",
    "End-to-end /verify latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

model_inference_seconds = Histogram(
    "aav_model_inference_seconds",
    "InsightFace inference time per frame",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
