from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import AutoModelForImageSegmentation

print("Downloading ViT...")
ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
print("ViT cached")

print("Downloading BLIP...")
BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP cached")

print("Downloading DETR...")
DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
print("DETR cached")

print("Downloading RMBG...")
AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
print("RMBG cached")

print("All models downloaded!")
