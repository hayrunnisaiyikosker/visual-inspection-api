# Visual Inspection API — Design Decisions

## 1. Model Selection

### 1.1 Image Classification: google/vit-base-patch16-224
**Why this model:**
Vision Transformer (ViT) was chosen over CNN-based alternatives because it provides
competitive top-3 accuracy on ImageNet with acceptable CPU inference time (~250ms).

**Alternatives considered:**
- microsoft/resnet-50: Faster (~120ms) but lower accuracy on fine-grained categories
- google/efficientnet-b0: Similar accuracy but less community support

**Known limitations:**
- Trained on ImageNet-1k (1000 general categories), not e-commerce specific
- Labels like "mailbag, postbag" instead of "tote bag" are expected — model sees
  visual similarity, not retail taxonomy
- Top-1 accuracy on domain-specific images will be lower than benchmark numbers

**CPU viability:** Confirmed. ~250ms on WSL2 CPU, acceptable for portfolio demo.

---

### 1.2 Image-to-Text: Salesforce/blip-image-captioning-base
**Why this model:**
BLIP (Bootstrapping Language-Image Pre-training) produces coherent product
descriptions in natural language. The "base" variant balances quality and CPU speed.

**Alternatives considered:**
- Salesforce/blip2-opt-2.7b: Higher quality but requires GPU, out of scope
- nlpconnect/vit-gpt2-image-captioning: Faster but lower quality descriptions
- microsoft/git-base: Similar quality, slightly slower

**Known limitations:**
- Generates generic descriptions; may not capture brand-specific details
- Max 50 new tokens per request (configured) to limit CPU time

**CPU viability:** Confirmed. ~1500ms on CPU, acceptable.

---

### 1.3 Object Detection: facebook/detr-resnet-50
**Why this model:**
DETR (Detection Transformer) uses attention mechanism to detect objects end-to-end
without anchor boxes. ResNet-50 backbone is the lightest DETR variant.

**Alternatives considered:**
- facebook/detr-resnet-101: Better accuracy but ~2x slower on CPU
- hustvl/yolos-small: Faster but less accurate on diverse objects

**Known limitations:**
- Trained on COCO dataset (80 categories); may misclassify products
  (e.g. a tote bag detected as "umbrella" due to visual shape similarity)
- Confidence threshold set at 0.7 to filter low-quality detections
- Bounding boxes returned as normalized coordinates (0-1 range) for
  frontend flexibility at any image resolution

**CPU viability:** Confirmed. ~1000ms on CPU.

---

### 1.4 Background Removal: briaai/RMBG-1.4
**Why this model:**
RMBG-1.4 by BRIA AI is purpose-built for product image background removal,
which directly matches our e-commerce use case. It runs on CPU without GPU dependency.

**Alternatives considered:**
- rembg (u2net): Popular library but heavier model, slower on CPU
- briaai/RMBG-2.0: Higher quality but requires GPU
- Stable Diffusion inpainting: Requires GPU, explicitly out of scope per brief

**Known limitations:**
- Input resized to 1024x1024 before inference; very large images lose detail
- Returns result as base64-encoded PNG with transparent background
- trust_remote_code=True required (documented security consideration)

**CPU viability:** Confirmed. ~1500ms on CPU.

---

## 2. Caching Strategy

**Implementation:** Redis with SHA-256 image hash as cache key
**TTL:** 86400 seconds (24 hours)

**Why SHA-256:**
- Deterministic: same image bytes always produce same hash
- Collision-resistant: two different images will not share a cache key
- Fast: hashing ~5MB image takes <1ms

**Cache hit flow:**
1. Compute SHA-256 of uploaded image bytes
2. Check Redis for key existence
3. If hit: return cached JSON without running any inference
4. If miss: run all 4 models, store result in Redis, return response

**Note:** Cache is keyed on exact bytes. Same product photo with different
compression or metadata will produce a different hash and miss the cache.
This is intentional — we cache exact matches only.

---

## 3. Async / Model Loading Strategy

**Model pre-loading:**
All 4 models are loaded once at application startup via FastAPI's @on_event("startup")
hook. This ensures:
- First request is as fast as subsequent requests
- No cold-start penalty per request
- Models stay in memory for the application lifetime

**Thread pool execution:**
Model inference is CPU-bound, not I/O-bound. FastAPI's async endpoints
delegate CPU-bound work to a thread pool executor (via run_in_executor)
to avoid blocking the event loop.

**Sequential vs parallel inference:**
The 4 models run sequentially in the unified /analyze endpoint. Running them
in parallel would require multi-threading with PyTorch, which introduces
GIL contention and potential memory issues on CPU. Sequential is safer and
more predictable for a portfolio demo.

**Trade-off:** ~4.3s total latency vs ~1.5s theoretical parallel. Acceptable
for demo purposes; production would use GPU or a task queue (Celery + Redis).

---

## 4. API Design

**Authentication:** API key via X-API-Key header (implemented in Phase 3)
- Appropriate for B2B seller platform use case
- Keys stored in PostgreSQL with monthly quota tracking

**Validation pipeline order:**
1. Content-type check (allowlist: JPEG, PNG, WebP)
2. File size check (max 5MB)
3. PIL verify() — confirms file is actually a valid image
4. Minimum dimensions check (50x50px)

Order matters: cheap checks (content-type, size) run before expensive ones (PIL).

---

## 5. Observability

**Per-request timing:** Each inference call is individually timed using
Python's time.time(). The processing_time_ms breakdown in the unified
response enables identifying bottlenecks without external tracing tools.

**Logging:** PostgreSQL stores image reference (hash), all inference results,
per-task timing, and timestamp. This enables offline analysis of model
performance over time.
