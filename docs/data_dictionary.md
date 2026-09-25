# Data dictionary

Every field this system stores, what it means, and - for the ones that touch a person - why
it is safe to keep. Ordered by how sensitive the store is.

A rule that applies to all of it: **there is no identity anywhere.** A worker is a label
inside one session (`Worker-017`), the numbering restarts every run, and nothing joins those
labels across sessions. That is what makes the rest of this table unremarkable.

---

## 1. `data/violations.jsonl` - the local violation log (append-only JSONL)

| Field | Type | Meaning | Personal data? |
|---|---|---|---|
| `time` | ISO-8601 UTC | when the line was written | no |
| `kind` | `person` \| `video` | per-worker record or per-clip summary | no |
| `source` | string | `image:<filename>`, `video:<filename>`, `camera:<index>` | only if the user names a file after a person |
| `person_id` | string | anonymous label within that one image/session | no - resets every run |
| `status` | string | `MISSING_HELMET`, `MISSING_VEST`, `POTENTIAL_VIOLATION`, ... | a judgement, not an identity |
| `missing` | list | which required items were absent | no |
| `confidence` | float 0-1 | detector confidence for that person box | no |
| `required` | list | the PPE required at that moment | no |
| `privacy_status` | enum | `NO_EVIDENCE` \| `FACE_BLUR_OK` \| `EVIDENCE_WITHHELD` | no |
| `snapshot` | path or null | **path only, never image bytes** | points at a masked file |

Video summary lines add `frames_analysed`, `frames_with_violation`, `violation_rate`,
`people_max`, `status_counts`.

**Retention:** the file is not rotated automatically. Stated as an open item in the threat
model rather than pretended.

---

## 2. `data/evidence/` - masked event snapshots (off by default)

JPEG files, written only when `PPE_STORE_SNAPSHOTS=true`, and only after the face band of
every detected person is pixelated (`shared/privacy.py`). If masking fails, no file is
written and the event is marked `EVIDENCE_WITHHELD`. Filenames are timestamps, not names.
`evidence_retention_days` (default 30) is defined but not yet enforced - see threat model §6.

---

## 3. `data/outbox.jsonl` - approved copilot alerts

| Field | Meaning |
|---|---|
| `time`, `priority` | when, and `HIGH` for a restricted-zone escalation |
| `action`, `reason`, `suggestion_id` | which allow-listed action, why, and the row it came from |
| `subject`, `source` | the anonymous worker label and where it was seen |
| `approved_by` | **the name the supervisor typed** - the one human-entered field in the system |

Nothing is transmitted from here. This file is the seam where a real site system would attach.

---

## 4. `data/server.db` (SQLite)

### `workers`
`worker_uid` (UUID, PK), `display_id` (`Worker-017`), `camera_id`, `session_id`,
`first_seen`, `last_seen`, `last_status`. The uniqueness constraint is
(camera, session, display_id) - so a label means nothing outside its session, by construction.

### `zones`
`zone_id` (PK), `zone_name`, `zone_type` (`GENERAL`/`CAUTION`/`RESTRICTED`), `camera_id`,
`polygon` (pixel coordinates), `required_ppe`, `authorization_required`, `priority`,
`active`, `config_version`, `updated_at`. Site configuration, no personal data.

### `events`
`event_id` (PK - **generated on the edge, so re-sending cannot duplicate a row**),
`device_id`, `camera_id`, `session_id`, `worker_uid`, `worker_display_id`, `zone_id`;
AI decision: `status`, `reasons`, `confidence`, `frames_in_window`, `frames_missing`,
`model_version`, `rules_version`, `occurred_at`, `received_at`, `payload_sha256`;
evidence: `privacy_status`, `evidence_path`, `evidence_expires_at`;
human decision summary: `review_status`.

The AI columns are never edited after insert, and the human decision lives in its own
columns - so "how often was the AI wrong?" stays answerable.

### `detections`
`detection_id` (PK), `event_id` (FK, cascade), `object_class`, `confidence`, `bbox`,
`assigned_to_worker`. Stored **only for event frames**, never for every frame - data
minimisation.

### `reviews`
`review_id` (PK), `event_id` (FK), `reviewer` (username, never the worker's name),
`decision` (`APPROVED`/`DISMISSED`), `comment`, `created_at`.

### `copilot_suggestions`
`suggestion_id` (PK), `created_at`, `source`, `subject`, `event_id`, `action_id` (always one
of `shared.copilot.ACTIONS`), `rule` (R2-R5), `reason`, `severity`, `evidence` (JSON),
`status` (`PENDING`/`APPROVED`/`REJECTED`), `decided_by`, `decided_at`, `note`, `outcome`.
Kept separate from `reviews` on purpose: a review judges whether the AI was right, a
suggestion proposes what to do. Keeping them apart is what makes the copilot's rejection
rate measurable.

### `audit_logs`
`audit_id` (PK), `occurred_at`, `actor` (`edge-01`, `copilot`, a reviewer's name, `system`),
`action` (`EVENT_CREATED`, `SUGGESTION_CREATED`, `SUGGESTION_APPROVED`, ...), `event_id`,
`details` (JSON), `prev_hash`, `row_hash`.
`row_hash = SHA-256(prev_hash + time + actor + action + event_id + details)`, so editing an
old row breaks every row after it. Append-only; `/api/v1/audit/verify` finds the first break.

---

## 4b. `configs/users.yaml` - sign-in accounts (gitignored)

| Field | Meaning |
|---|---|
| `username` | the account name typed at sign-in |
| `role` | `viewer` / `supervisor` / `admin` |
| `display_name` | what appears in the audit trail |
| `salt`, `hash` | PBKDF2-HMAC-SHA256, 200 000 iterations, per-user salt |

**No password is ever stored**, here or anywhere else, and the file is per-machine rather
than source. A missing file means nobody can sign in - so the site can be viewed but nothing
can be approved, which is the safe direction.

## 5. Dataset (`datasets/ppe4/`)

Roboflow Universe *Construction Site Safety* (CC BY 4.0). Labels are YOLO-format text files:
`class cx cy w h`, normalised, class order fixed as `0 person, 1 helmet, 2 vest, 3 mask`.
`subset_manifest.json` records exactly which images were selected and with which seed;
`stats.json` records the per-class counts. Splits are grouped by the *original photo* so
augmented copies of one photo cannot straddle train and validation.

The images are public, licensed, and were never collected by this project. No consented
private site footage is used, and none should be added without a consent process.

---

## 6. What is deliberately NOT stored anywhere

Faces, names, employee numbers, face embeddings, biometric templates, raw camera frames,
timelines of where a person was, dwell time, or any identifier that survives a session.
Adding any of them would be a change of purpose, not a feature.
