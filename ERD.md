# Visual Inspection API — Entity Relationship Diagram

## Overview

The database serves two purposes:
1. Request logging — every inference call is recorded for observability
2. API key management — keys, rate limits, and monthly quotas (Phase 3)

---

## Tables

### api_keys
Stores API keys issued to seller platforms.

| Column           | Type         | Constraints          | Description                              |
|------------------|--------------|----------------------|------------------------------------------|
| id               | UUID         | PK, default uuid     | Primary key                              |
| key_hash         | VARCHAR(64)  | UNIQUE, NOT NULL     | SHA-256 hash of the raw API key          |
| name             | VARCHAR(100) | NOT NULL             | Label e.g. Noon Integration              |
| monthly_quota    | INTEGER      | NOT NULL, default 1000 | Max inference calls per calendar month |
| calls_this_month | INTEGER      | NOT NULL, default 0  | Resets on 1st of each month              |
| is_active        | BOOLEAN      | NOT NULL, default true | Soft-disable without deletion           |
| created_at       | TIMESTAMP    | NOT NULL, default now() | Creation timestamp                    |
| last_used_at     | TIMESTAMP    | NULLABLE             | Timestamp of last successful request     |

---

### inference_logs
One row per inference request. Stores results and per-task timing.

| Column                | Type        | Constraints             | Description                            |
|-----------------------|-------------|-------------------------|----------------------------------------|
| id                    | UUID        | PK, default uuid        | Primary key                            |
| api_key_id            | UUID        | FK to api_keys.id       | Which key made this request            |
| image_hash            | VARCHAR(64) | NOT NULL, indexed       | SHA-256 of uploaded image bytes        |
| classification_result | JSONB       | NULLABLE                | Full classification response           |
| description_result    | JSONB       | NULLABLE                | Full description response              |
| detection_result      | JSONB       | NULLABLE                | Full detection response                |
| background_removed    | BOOLEAN     | NOT NULL, default true  | Whether bg removal ran                 |
| classification_ms     | FLOAT       | NULLABLE                | Inference time in milliseconds         |
| description_ms        | FLOAT       | NULLABLE                | Inference time in milliseconds         |
| detection_ms          | FLOAT       | NULLABLE                | Inference time in milliseconds         |
| background_removal_ms | FLOAT       | NULLABLE                | Inference time in milliseconds         |
| total_ms              | FLOAT       | NULLABLE                | Total pipeline time in milliseconds    |
| cache_hit             | BOOLEAN     | NOT NULL, default false | Was this response served from cache?   |
| endpoint              | VARCHAR(50) | NOT NULL                | Which endpoint was called              |
| created_at            | TIMESTAMP   | NOT NULL, default now() | Request timestamp                      |

---

## Relationships

api_keys (1) --------< inference_logs (many)
   id                      api_key_id (FK)

One API key can have many inference log entries.
Each log entry belongs to exactly one API key.

---

## Indexes

| Table          | Column       | Reason                                  |
|----------------|--------------|-----------------------------------------|
| inference_logs | image_hash   | Fast cache lookup by image hash         |
| inference_logs | created_at   | Time-range queries for analytics        |
| inference_logs | api_key_id   | Filter logs by key                      |
| api_keys       | key_hash     | Fast API key lookup on every request    |

---

## Design Notes

- JSONB is used for inference results so the structure can evolve
  without requiring schema migrations
- image_hash links the DB log to the Redis cache entry
- cache_hit column allows measuring cache effectiveness over time
- Monthly quota reset is handled by the application layer on each request,
  not by a DB trigger, for simplicity
- Raw API keys are never stored; only their SHA-256 hash is persisted
