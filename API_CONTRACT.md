# Visual Inspection API — API Contract
**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Authentication:** `X-API-Key` header (Phase 3)

---

## General Rules

- All endpoints accept `multipart/form-data` with a single `file` field
- Allowed file types: JPEG, PNG, WebP
- Maximum file size: 5 MB
- Minimum image dimensions: 50×50px
- All responses are `application/json`
- Errors follow a consistent shape: `{"error": "...", "detail": "..."}`

---

## Endpoints

### 1. Health Check
**GET** `/health`

No authentication required.

**Response 200:**
```json
{
  "status": "ok"
}
```

---

### 2. Image Classification
**POST** `/analyze/classify`

Accepts a product image and returns the predicted category with confidence scores.  
Model: `google/vit-base-patch16-224`

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file  | file | Yes      | Product image (JPEG/PNG/WebP, max 5MB) |

**Response 200:**
```json
{
  "top_prediction": "running shoe",
  "confidence": 0.94,
  "top_3": [
    {"label": "running shoe", "confidence": 0.94},
    {"label": "sneaker",      "confidence": 0.04},
    {"label": "sandal",       "confidence": 0.02}
  ]
}
```

**Error Responses:**
| Code | Reason |
|------|--------|
| 413  | File exceeds 5MB |
| 415  | Unsupported file type |
| 422  | Invalid image or dimensions too small |

---

### 3. Image Description
**POST** `/analyze/describe`

Generates a natural-language product description from a product image.  
Model: `Salesforce/blip-image-captioning-base`

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file  | file | Yes      | Product image (JPEG/PNG/WebP, max 5MB) |

**Response 200:**
```json
{
  "raw": "a green and black tote bag with a black and white design",
  "cleaned": "A green and black tote bag with a black and white design"
}
```

---

### 4. Object Detection
**POST** `/analyze/detect`

Detects all objects in the image and returns bounding boxes as normalized coordinates.  
Model: `facebook/detr-resnet-50`  
Confidence threshold: 0.7

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file  | file | Yes      | Product image (JPEG/PNG/WebP, max 5MB) |

**Response 200:**
```json
{
  "detected_objects": [
    {
      "label": "handbag",
      "confidence": 0.94,
      "bbox": {
        "x_min": 0.19,
        "y_min": 0.13,
        "x_max": 0.82,
        "y_max": 0.88
      }
    }
  ],
  "total_objects": 1
}
```

**Bounding box coordinate system:**  
All values are normalized to [0, 1] relative to image width and height.  
This allows the frontend to overlay boxes on any display resolution.

---

### 5. Background Removal
**POST** `/analyze/remove-background`

Removes the image background and returns the product isolated on a transparent background.  
Model: `briaai/RMBG-1.4`

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file  | file | Yes      | Product image (JPEG/PNG/WebP, max 5MB) |

**Response 200:**
```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

Output is always PNG with an alpha (transparency) channel,  
encoded as a base64 data URI ready for direct use in an img tag.

---

### 6. Unified Analysis (Signature Endpoint)
**POST** `/analyze/`

Runs all four AI tasks in a single call and returns a unified structured response.  
This is the primary endpoint for seller platform integrations.

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file  | file | Yes      | Product image (JPEG/PNG/WebP, max 5MB) |

**Response 200:**
```json
{
  "classification": {
    "top_prediction": "running shoe",
    "confidence": 0.94,
    "top_3": [
      {"label": "running shoe", "confidence": 0.94},
      {"label": "sneaker",      "confidence": 0.04},
      {"label": "sandal",       "confidence": 0.02}
    ]
  },
  "description": {
    "raw": "a red running shoe on a white surface",
    "cleaned": "A red running shoe on a white surface"
  },
  "detection": {
    "detected_objects": [
      {
        "label": "shoe",
        "confidence": 0.98,
        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
      }
    ],
    "total_objects": 1
  },
  "background_removed": "data:image/png;base64,...",
  "processing_time_ms": {
    "classification_ms": 247.0,
    "description_ms": 1566.0,
    "detection_ms": 1006.0,
    "background_removal_ms": 1478.0,
    "total_ms": 4299.0
  }
}
```

**Caching behavior:**  
If the same image (identified by SHA-256 hash) is submitted within 24 hours,  
the cached result is returned immediately without re-running inference.  
Cache hits are served in under 10ms.

---

## Error Response Format

All errors return a consistent JSON shape:

```json
{
  "error": "File too large",
  "detail": "File size 6.2 MB exceeds the 5 MB limit"
}
```

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Bad request |
| 401 | Missing or invalid API key |
| 413 | File too large (> 5MB) |
| 415 | Unsupported media type |
| 422 | Validation error (bad image, too small) |
| 429 | Rate limit or monthly quota exceeded |
| 500 | Internal server error |
