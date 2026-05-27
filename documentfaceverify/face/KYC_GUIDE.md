# KYC Verification — Non-technical Guide

A short, plain-language explanation of what this system does, what it
reads off each ID document, and how it makes a pass / review / fail
decision.

---

## What the system does in one sentence

A user uploads a photo of an Indian government ID (Aadhaar, PAN, Voter
ID, Driving Licence, or Passport), takes a quick live selfie on their
webcam, and the system answers one question: **is the person on the
camera the same person on the document?**

It also pulls the useful details off the document (name, date of birth,
etc.) so you can fill the rest of your form with one upload.

---

## The flow, step by step

1. **Upload the ID.** User picks a JPG, PNG, or PDF of the document and
   tells us which type it is (Aadhaar / PAN / Voter ID / Driving Licence
   / Passport).
2. **We read the document.** Two things happen at the same time:
   - We pull the face photo off the document.
   - We read the printed text (using OCR if it's a scan) and extract
     the labelled fields — name, DOB, ID number, address, etc. — into
     a structured report.
3. **User takes a live selfie.** The browser captures a short sequence
   of webcam frames. We use those to confirm the user is physically
   present and not just holding up a photo.
4. **We compare faces.** The face from the document is compared with
   the live face. A match score from 0 to 100 comes out.
5. **A decision is made.**
   - **VERIFIED** — high confidence it's the same person. Auto-approve.
   - **MANUAL REVIEW** — borderline; a human should look at it.
   - **REJECTED** — low confidence; deny the application.

The full report (the extracted fields + the decision) is returned to
your application and shown on the result screen.

---

## What we read off each document

The fields below are what the system tries to extract. If a field can't
be found cleanly, it's simply left out — the system never invents data.

### Aadhaar Card
- Full name
- Date of birth
- Gender
- Aadhaar number — **always shown masked** as `XXXX-XXXX-1234`. The
  full 12 digits never leave the parser.

### PAN Card
- Full name
- Father's name
- Date of birth
- PAN number — **shown masked** as `ABCXX1234F`. The middle digits are
  hidden.

### Voter ID (EPIC)
- Full name
- Father's / husband's name
- Gender
- Date of birth or age
- Address
- EPIC / identity card number

### Driving Licence
- Licence number
- Full name
- Father's name
- Date of birth
- Gender
- Date of issue
- Validity (expiry)
- Address

### Passport
- Surname
- Given name(s)
- Date of birth
- Gender
- Address
- Place of issue
- Date of issue
- Date of expiry
- Mother's name
- Legal guardian
- Name of spouse
- File number
- Passport number — **shown masked** as `MXXX4567`.

---

## How the match score is decided

Every document type has its own pass mark. This is deliberate. The
photo printed on an Aadhaar card is tiny and heavily compressed; the
photo on a passport's data page is much larger and higher quality. It
would be unfair to apply the same bar to both. The bands below come
from a calibrated safe-values table.

| Document | Auto-approve if score ≥ | Manual review if score ≥ | Below review = reject |
|---|---|---|---|
| **Aadhaar** | 46% | 43% | below 43% |
| **PAN** | 52% | 49% | below 49% |
| **Driving Licence** | 56% | 53% | below 53% |
| **Voter ID** | 43% | 40% | below 40% |
| **Passport** | 63% | 50% | below 50% |

The score is a percentage from 0 to 100. Higher means the system is
more confident the two faces are the same person.

### Why the thresholds also adjust for age

ID photos are often years out of date. The system estimates how old the
card photo is (using the face on the document and a live face), and if
there's a big age gap it slightly relaxes the bar — up to about 10
points. So a 40-year-old verifying against a 15-year-old Aadhaar photo
isn't punished for looking older now.

### Decision tiers — what they mean to your business

- **VERIFIED** — proceed automatically. The system is confident.
- **MANUAL REVIEW** — don't reject the user; send the case to your ops
  team to eyeball it. Often a real customer who just has a poor or
  old document photo.
- **REJECTED** — high likelihood the person on the camera isn't the
  person on the ID. Refuse the application.

---

## How we keep data safe

- The uploaded card image is **never written to disk**. Once we've
  extracted the face crop and the text, the raw image is dropped.
- Sensitive numbers (Aadhaar, PAN, passport) are **masked before they
  enter the system's memory** or logs. The unmasked value is not
  available anywhere except in the parsing function itself, which
  immediately throws it away.
- Each verification session lives for **10 minutes max** in our cache
  and is **deleted the instant the verification finishes**.
- All connections must be over HTTPS in real deployments. The bundled
  development setup uses plain HTTP for `localhost` only.
- Every decision is written to an **audit log** that contains scores
  and reason codes — but no biometrics, no images, no embeddings.
  This is what compliance teams can review later.

---

## What can go wrong (be honest about the limits)

- **Old or stained card photos.** Aadhaar prints fade and smudge. A
  legitimate user can land in MANUAL REVIEW just because their card
  scans poorly. That's the right outcome — don't auto-approve, don't
  auto-reject.
- **Bad camera lighting.** A washed-out live selfie can lower the
  match score even when it's the same person. We tell the user to
  improve lighting in the review reason.
- **Liveness can be fooled by a video on a screen.** We have basic
  protection (head motion challenges, a moire-pattern advisory) but
  a sophisticated video replay attack can still get through. A real
  production deployment should add a trained anti-spoof model — that
  work is scoped but not yet built.
- **The text parser is best-effort.** Driving licence formats vary
  wildly by state, and some fields will be left empty. That doesn't
  mean the verification failed — only that we couldn't find that
  particular field on this document.
- **This is not UIDAI authentication.** We match the face printed on
  the card against the live face. We do not call UIDAI's central
  database. For a regulated KYC check in India, the customer must
  separately complete an Aadhaar OTP or biometric authentication
  through a licensed AUA / KUA. That is a legal step, not a code
  step.

---

## How to tune the bands

If your business is willing to accept more manual reviews in exchange
for fewer false approvals, raise the verify percentage for that
document. If you want more auto-approvals and have ops bandwidth for
mistakes, lower it. Each number is a setting:

| Setting (in `.env`) | What it does |
|---|---|
| `AADHAAR_VERIFY_PCT` | Auto-approve cutoff for Aadhaar |
| `AADHAAR_REVIEW_PCT` | Reject below this for Aadhaar |
| `PAN_VERIFY_PCT` / `PAN_REVIEW_PCT` | Same, for PAN |
| `DL_VERIFY_PCT` / `DL_REVIEW_PCT` | Same, for Driving Licence |
| `VOTER_VERIFY_PCT` / `VOTER_REVIEW_PCT` | Same, for Voter ID |
| `PASSPORT_VERIFY_PCT` / `PASSPORT_REVIEW_PCT` | Same, for Passport |

Changes take effect on the next service restart. No code changes
needed.

---

## In short

1. **Customer uploads an ID + takes a selfie.**
2. **We read the ID's text into a structured report.**
3. **We compare faces and produce a score from 0–100.**
4. **The score is checked against a per-document bar** to land on
   `VERIFIED`, `MANUAL REVIEW`, or `REJECTED`.
5. **The decision + extracted report** is returned to your system.
6. **No images are kept, sensitive numbers are masked, and the
   session is deleted after one attempt.**
