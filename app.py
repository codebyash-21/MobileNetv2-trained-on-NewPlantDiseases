import json
import os
import urllib.request
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "best_mnv2_v2.keras")
LABELS_PATH = os.path.join(HERE, "labels.json")
IMG_SIZE = 160

# Auto-download model from Hugging Face if not present locally
MODEL_URL = "https://huggingface.co/ashikaasriarun/MobileNet-V2-NewPlantDisease/resolve/main/best_mnv2_v2.keras"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found locally. Downloading from Hugging Face (~28 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = json.load(open(LABELS_PATH))
print(f"Loaded {len(labels)} classes.")


def pretty(name):
    if "___" in name:
        crop, _, disease = name.partition("___")
        return f"{crop.replace('_', ' ')} - {disease.replace('_', ' ')}"
    return name.replace("_", " ")


def predict(img):
    if img is None:
        return {}, "Upload an image."
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32)[np.newaxis, ...]
    x = preprocess_input(x)
    probs = model.predict(x, verbose=0)[0]
    top5 = probs.argsort()[-5:][::-1]
    result = {pretty(labels[i]): float(probs[i]) for i in top5}
    best = top5[0]
    verdict = f"{pretty(labels[best])} ({probs[best]*100:.1f}% confident)"
    return result, verdict


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload leaf image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top predictions"),
        gr.Textbox(label="Verdict"),
    ],
    title="Plant Disease Detector (MobileNetV2 + New Plant Diseases)",
    description="MobileNetV2 trained on the New Plant Diseases Dataset (~87k augmented images, 99.00% accuracy). Upload a leaf image to get a disease prediction.",
)

if __name__ == "__main__":
    demo.launch()
