FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y     libgl1     libglib2.0-0     libpq-dev     gcc     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python3 -c "from transformers import ViTForImageClassification, ViTImageProcessor; ViTImageProcessor.from_pretrained("google/vit-base-patch16-224"); ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"); print("ViT cached")"

RUN python3 -c "from transformers import BlipProcessor, BlipForConditionalGeneration; BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"); BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"); print("BLIP cached")"

RUN python3 -c "from transformers import DetrImageProcessor, DetrForObjectDetection; DetrImageProcessor.from_pretrained("facebook/detr-resnet-50"); DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50"); print("DETR cached")"

RUN python3 -c "from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True); print("RMBG cached")"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
