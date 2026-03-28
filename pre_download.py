import os
from transformers import pipeline

print("Pre-loading AI models...")
# Download and cache the model during build time
pipeline("image-classification", model="umm-maybe/AI-image-detector")
print("Model pre-loaded successfully.")
