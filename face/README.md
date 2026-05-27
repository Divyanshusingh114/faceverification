# Aadhaar Face Verification

Match a live webcam capture against the photo printed on an Aadhaar card,
with an active liveness check. FastAPI backend, single-page web frontend,
InsightFace + ArcFace on CPU.

This repository started as a prototype and has since been hardened for
deployment: Redis-backed sessions with atomic consume, API-key auth,
rate limiting, security headers, magic-byte image validation, structured
JSON logs, Prometheus metrics, an append-only audit log, Docker image
with the model baked in, and CI.

It is still **not UIDAI authentication**. It matches the printed card
photo, not UIDAI's enrolment record. A real KYC integration in India
requires going through a licensed AUA/KUA or Sub-AUA. See "Compliance
scope" below.

## Pipeline

```
 Upload Aadhaar  --->  Extract reference face   (InsightFace detect + ArcFace)
                          |
 Live webcam     --->  Random liveness challenge (turn / nod / lean in)
                          |
                       Verify movement happened  (head-pose proxies)
                          |
                       Match faces               (cosine similarity)
                          |
                       Tiered decision           VERIFIED / MANUAL_REVIEW / REJECTED
```

| Stage | How |
|---|---|
| Reference face | `buffalo_l` detects the face on the card; a quality gate rejects small, low-confidence, or undecodable crops. |
| Liveness | Server picks a random challenge. The browser captures frames; the backend verifies the movement using yaw / pitch / width proxies derived from the 5 face keypoints. |
| Face match | 512-d ArcFace embeddings. Frontal live frames are averaged and renormalised, then compared to the reference by cosine similarity. |
| Decision | Tiered: `VERIFIED` (auto-approve), `MANUAL_REVIEW` (step-up, e.g. Aadhaar OTP), `REJECTED`. Thresholds relax with the estimated age gap between the two photos. |

## Project structure

```
aav/
├── main.py              # FastAPI app, lifespan, middleware, route handlers
├── settings.py          # Pydantic Settings - all env vars, cached
├── core/
│   ├── audit.py         # Append-only JSONL audit log (no biometrics)
│   ├── errors.py        # Stable error codes + APIError class
│   ├── logging.py       # structlog JSON setup
│   ├── metrics.py       # Prometheus counters / histograms
│   └── security.py      # API key, rate limiter, headers, image validation
├── pipeline/
│   ├── aadhaar.py       # Card decode, face extraction, quality gate
│   ├── face_engine.py   # InsightFace buffalo_l wrapper (singleton)
│   ├── liveness.py      # Challenge logic + head-pose proxies + replay heuristic
│   └── verify.py        # Cosine match + age-gap adaptive thresholding
├── storage/
│   └── sessions.py      # Redis session store with atomic GETDEL consume()
└── web/static/
    └── index.html       # Single-page frontend (upload + webcam + result)

scripts/download_models.py    # Bake buffalo_l into the Docker image
tests/                        # Pytest suite (uses fakeredis, stubs the model)
Dockerfile                    # Multi-stage build, non-root user, tini, model baked
docker-compose.yml            # app + redis with healthchecks
.github/workflows/ci.yml      # Ruff, mypy, pytest, docker build
```

## Configuration

All knobs are environment variables, loaded by `aav/settings.py` (also
readable from a `.env` file). `.env.example` is the source of truth for
defaults; the table below matches it.

| Variable | Default | Purpose |
|---|---|---|
| `APP_NAME` | `aadhaar-face-verify` | Service name (logs only). |
| `ENV` | `dev` | One of `dev` / `staging` / `prod`. |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |
| `HOST` | `0.0.0.0` | Bind address (used when running uvicorn directly). |
| `PORT` | `8000` | Bind port. |
| `CORS_ORIGINS` | _(empty)_ | Comma-separated origins. Empty disables CORS. |
| `API_KEYS` | _(empty)_ | Comma-separated allowed keys for `X-API-Key`. Empty disables auth (dev only). |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL. |
| `SESSION_TTL_SECONDS` | `600` | TTL for verification sessions in Redis. |
| `MAX_UPLOAD_BYTES` | `5242880` | Hard cap on Aadhaar image size (5 MB). |
| `MAX_FRAMES` | `40` | Max frames accepted per `/verify` call. |
| `MAX_FRAME_BYTES` | `1048576` | Per-frame size cap (1 MB raw). |
| `RATE_LIMIT_UPLOAD` | `10/minute` | slowapi rate limit on `/api/upload`. |
| `RATE_LIMIT_VERIFY` | `20/minute` | slowapi rate limit on `/api/verify`. |
| `DET_SIZE` | `640` | InsightFace detector input size. |
| `USE_GPU` | `false` | If true, prefer `CUDAExecutionProvider`. |
| `BASE_VERIFIED` | `0.50` | Cosine threshold for auto-approve (legacy `decide()`, kept for reference). |
| `BASE_REVIEW` | `0.36` | Cosine threshold for manual review (legacy). |
| `MAX_RELAX` | `0.10` | Max cosine-space relaxation from age-gap adaptation. |
| `RELAX_PER_YEAR` | `0.005` | Cosine relaxation added per year of estimated age gap. |
| `PCT_VERIFIED` | `75.0` | **Public** decision: `score_pct >= this` → VERIFIED. |
| `PCT_REVIEW` | `50.0` | `score_pct >= this` (but below verified) → MANUAL_REVIEW; below → REJECTED. |
| `PCT_MAX_RELAX` | `10.0` | Max age-gap relaxation in pct units. |
| `PCT_RELAX_PER_YEAR` | `0.5` | Relaxation added per year of estimated age gap. |
| `MIN_FACE_PX` | `48` | Reject reference faces smaller than this on the shortest side. |
| `MIN_DET_SCORE` | `0.50` | Reject reference faces below this detector confidence. |
| `BLUR_VAR_WARN` | `12.0` | Variance-of-Laplacian threshold for the advisory blur flag. |
| `BLUR_NORM_WIDTH` | `160` | Crop width used to size-normalise the blur score. |
| `TURN_DELTA` | `0.18` | Min yaw range for a `turn` challenge to pass. |
| `NOD_DELTA` | `0.12` | Min pitch range for a `nod` challenge to pass. |
| `CLOSER_RATIO` | `1.22` | Min bbox-width ratio for a `closer` challenge to pass. |
| `FRONTAL_YAW` | `0.12` | Yaw magnitude under which a frame is considered frontal. |
| `MIN_FACE_FRAMES` | `0.55` | Fraction of frames that must contain exactly one face. |
| `METRICS_ENABLED` | `true` | Expose `/metrics` for Prometheus. |
| `AUDIT_LOG_PATH` | `/var/log/aadhaar-verify/audit.log` | Path for the append-only decision log. |

## Run locally (Docker)

```bash
cp .env.example .env
# At minimum, set API_KEYS=somekey and CORS_ORIGINS=http://localhost:8000
docker compose up --build
```

`docker-compose.yml` brings up Redis and the app together, wired with
healthchecks. The app image bakes the `buffalo_l` model during build, so
the first request is warm.

Endpoints exposed on port 8000:

- `GET /` — single-page frontend
- `POST /api/document` — upload any supported KYC document (aadhaar / pan / voter_id / driving_licence / passport), parse fields, return session id
- `POST /api/upload` — legacy alias for `/api/document`; `doc_type` defaults to `aadhaar`
- `GET /api/challenge/{sid}` — fetch a random liveness challenge
- `POST /api/verify/{sid}` — submit webcam frames, get tiered decision + parsed document report
- `GET /api/healthz` — liveness probe (process is up)
- `GET /api/readyz` — readiness probe (Redis reachable, model loaded)
- `GET /metrics` — Prometheus exposition

Browsers require HTTPS or `localhost` for camera access.

## Run locally (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Start Redis separately, e.g.:
docker run --rm -p 6379:6379 redis:7-alpine

cp .env.example .env   # then edit
uvicorn aav.main:app --reload
```

First run downloads the `buffalo_l` pack (~300 MB) under
`~/.insightface`. CPU-only inference works out of the box.

## API reference

All `/api/*` endpoints require `X-API-Key: <key>` when `API_KEYS` is set.
Errors always return `{"ok": false, "code": "<slug>", "message": "..."}`
with codes drawn from `aav/core/errors.py`.

### `POST /api/document` (and legacy `POST /api/upload`)

Multipart form upload.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `file` | file | yes | — | JPEG / PNG / WebP / PDF, max 5 MB. |
| `doc_type` | string | no | `aadhaar` | One of `aadhaar`, `pan`, `voter_id`, `driving_licence`, `passport`. |

The server runs PyMuPDF on the bytes (with Tesseract OCR fallback for
image inputs and image-only PDFs), parses the doc-type-specific fields,
extracts the largest face, and stashes the reference embedding in Redis.

Response 200:
```json
{
  "ok": true,
  "session_id": "hex-uuid",
  "document": {
    "doc_type": "pan",
    "fields": {
      "pan_number_masked": "ABCXX1234F",
      "name": "RAHUL VERMA",
      "fathers_name": "SUNIL VERMA",
      "dob": "02/11/1990"
    },
    "parse_confidence": 1.0
  },
  "reference_quality": {
    "face_px": 220,
    "det_score": 0.97,
    "est_age": 31.0
  },
  "note": "Reference face accepted. Proceed to live verification."
}
```

Sensitive identifiers (Aadhaar 12-digit, PAN, passport number) are
**masked at parse time** by `aav/pipeline/parsers.py` — the unmasked value
never enters session state, logs, or the API response.

Possible error codes: `invalid_image`, `image_too_large`,
`unsupported_doc_type`, `reference_rejected` (reason =
`no_face_found_on_document` / `face_too_small` / `low_detection_confidence` /
`could_not_decode` / `could_not_render_page` / `unsupported_doc_type`),
`unauthorized`, `rate_limited`.

### `GET /api/challenge/{sid}`

Response 200:
```json
{
  "challenge_id": "turn",
  "instruction": "धीरे-धीरे अपना सिर एक तरफ घुमाएं, और फिर वापस बीच में लाएं।"
}
```

`challenge_id` is one of `turn`, `nod`, `closer`. Error codes:
`session_not_found_or_expired`, `unauthorized`.

### `POST /api/verify/{sid}`

JSON body:
```json
{
  "frames": ["data:image/jpeg;base64,...", "..."],
  "mode": "live"
}
```

`frames` is an array of data-URL encoded JPEGs from the webcam. `mode`
defaults to `live`; use `upload` only for an offline image flow that
skips liveness.

Response 200:
```json
{
  "challenge_id": "turn",
  "liveness": {
    "passed": true,
    "reason": "ok",
    "evidence": {"face_frame_ratio": 0.93, "yaw_range": 0.27}
  },
  "decision": {
    "tier": "VERIFIED",
    "score_pct": 82.1,
    "similarity": 0.8210,
    "age_gap_years": 4.0,
    "threshold_verified_pct": 73.0,
    "threshold_review_pct": 48.0,
    "relaxation_applied_pct": 2.0,
    "frontal_frames_used": 7,
    "passive_replay_score": 0.41,
    "passive_replay_note": "advisory heuristic only - high values may indicate a screen/print replay"
  },
  "document": {
    "doc_type": "pan",
    "fields": {"pan_number_masked": "ABCXX1234F", "name": "RAHUL VERMA"},
    "parse_confidence": 1.0
  }
}
```

**Tiers** are derived from `score_pct` (= `max(0, cosine) * 100`) against
the age-relaxed thresholds:

| Band | Default cutoff | Tier |
|---|---|---|
| `score_pct >= PCT_VERIFIED - relax` | ≥ 75 | `VERIFIED` |
| `score_pct >= PCT_REVIEW - relax` | 50 – 74 | `MANUAL_REVIEW` |
| else | < 50 | `REJECTED` |

The session is consumed atomically on this call: a second request with
the same `sid` returns `session_not_found_or_expired`. Other error
codes: `no_frames_received`, `too_many_frames`, `frame_too_large`,
`request_a_challenge_first`, `unauthorized`, `rate_limited`.

When liveness fails, the response carries `decision.tier: "REJECTED"`
with `cause: "liveness_failed"`. When no usable frontal frame is found,
`tier: "MANUAL_REVIEW"` with `cause: "no_frontal_frame_for_match"`.

### `GET /api/healthz`

Returns `{"status": "ok"}`. No dependencies checked — process liveness
only.

### `GET /api/readyz`

Returns 200 `{"status": "ready", "redis": true}` when Redis is
reachable, otherwise 503 with code `service_not_ready`.

### `GET /metrics`

Prometheus exposition (when `METRICS_ENABLED=true`).

## Security posture

Enforced in code, not just in docs:

- **API-key auth** on every `/api/*` route via `X-API-Key`. When
  `API_KEYS` is empty (dev only), the dependency is a no-op.
- **Per-IP rate limits** via slowapi on `/api/upload` and `/api/verify`.
- **Magic-byte upload validation** — only JPEG / PNG / WebP / PDF accepted.
  SVGs, archives and zip-bombs are rejected before any decode.
- **Sensitive-number masking at parse time** — Aadhaar, PAN, and passport
  numbers are masked by the parsers themselves, so the unmasked value
  cannot enter session state or response bodies.
- **Hard size caps** — `MAX_UPLOAD_BYTES`, `MAX_FRAMES`, `MAX_FRAME_BYTES`.
- **Security headers** on every response: `X-Content-Type-Options`,
  `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`
  (camera self only), `Strict-Transport-Security`, and a restrictive
  default CSP.
- **CORS allowlist** via `CORS_ORIGINS` (empty disables cross-origin).
- **Redis-backed sessions** with TTL enforced by Redis itself.
- **Atomic `consume()`** — the verify handler reads and deletes the
  session in a single pipelined transaction, so a session embedding
  cannot be reused under concurrent requests.
- **No biometrics in logs.** Embeddings, raw images and face crops never
  reach `structlog` nor the audit log.
- **Container runs as non-root** (uid 10001) with `tini` as PID 1.

## Observability

- **Structured JSON logs** via `structlog` (`aav/core/logging.py`).
  Every request emits stable event names: `upload_accepted`,
  `upload_rejected`, `verify_complete`, lifecycle events, etc.
- **Prometheus** — HTTP metrics from `prometheus-fastapi-instrumentator`
  at `/metrics`, plus domain counters defined in `aav/core/metrics.py`:
  - `aav_decisions_total{tier}` — VERIFIED / MANUAL_REVIEW / REJECTED
  - `aav_liveness_failures_total{reason}`
  - `aav_upload_rejections_total{reason}`
  - `aav_verify_seconds` histogram
  - `aav_model_inference_seconds` histogram
- **Audit log** at `AUDIT_LOG_PATH` (default
  `/var/log/aadhaar-verify/audit.log`, mounted as a docker volume).
  One JSON line per decision: sid, timestamp, tier, scores, reasons.
  Contains zero biometrics — safe to ship to a SIEM with its own
  retention policy.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite stubs InsightFace and uses `fakeredis`, so it runs without
the model pack or a live Redis.

| Module | Focus |
|---|---|
| `tests/test_aadhaar.py` | Quality-gate behaviour: small face reject, low-detection-score reject, advisory blur flag. |
| `tests/test_app_smoke.py` | HTTP error mapping, healthz, unauthorized paths. Stubs lifespan deps. |
| `tests/test_liveness.py` | `random_challenge`, `verify_challenge` pass/fail for turn / nod / closer, frame-ratio guard. |
| `tests/test_security.py` | Magic-byte validation, frame/upload size limits. |
| `tests/test_sessions.py` | Redis session store: create, set_challenge, consume atomicity, TTL, corrupt embedding handling. |
| `tests/test_verify.py` | Cosine similarity, embedding averaging, age relaxation curve, tiered `decide()`. |

CI (`.github/workflows/ci.yml`) runs ruff, mypy, pytest, and a Docker
build on every push and PR.

## Calibration and known limits

What is still required before this is real KYC:

- **Thresholds need calibration.** `BASE_VERIFIED` (0.50) and
  `BASE_REVIEW` (0.36) are sane starting points for ArcFace
  (`w600k_r50`) cosine similarity on unit vectors, but you must measure
  genuine-vs-impostor scores on your own data and pick cutoffs for a
  target false-accept rate. Same applies to `MAX_RELAX` and
  `RELAX_PER_YEAR`.
- **Aadhaar photos are the dominant source of false rejects** — they
  are small, heavily compressed, and often 5–15 years old. The quality
  gate rejects the worst uploads early, and the tiered decision routes
  borderline matches to manual review instead of hard-rejecting; that
  posture is correct but it does not raise true-accept rate. The single
  biggest accuracy lever is fine-tuning ArcFace on a cross-age dataset
  (CACD / AgeDB / FG-NET).
- **Active liveness covers presentation attacks** (card, printed photo,
  still image) but a high-quality **video replay on a screen** can
  still defeat it. The FFT `passive_replay_score` is a weak advisory
  heuristic only. Production systems should add a trained anti-spoof
  model (MiniFASNet, Silent-Face, etc.) or IR / depth hardware.
- **Replay-attack scoring is not calibrated.** Do not hard-block on
  `passive_replay_score` without measuring it on genuine and attack
  samples from your own deployment.
- **Single-language UI.** The challenge instructions in `liveness.py`
  are Hindi only; localise before shipping outside that audience.

## Compliance scope

This service is **not** UIDAI authentication. It performs a 1:1 face
match between a live capture and the photo printed on the Aadhaar card
the user uploads. It does **not**:

- query UIDAI,
- read the 12-digit Aadhaar number (and OCR/masking should be done
  upstream if you ever need the number),
- store any biometric, even briefly, outside of an in-memory Redis
  session that auto-expires.

A real Aadhaar-based identity check in India requires going through a
licensed AUA / KUA or Sub-AUA with the appropriate agreements,
infrastructure, and audit obligations. Treat this repository as the
*face-matching and liveness layer* of such a pipeline, not as the
identity check itself.
