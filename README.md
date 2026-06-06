# Plant Disease Detector — MobileNetV2 + New Plant Diseases

Deep learning model for plant disease detection, trained on the largest augmented PlantVillage dataset.

## Model

- **Architecture:** MobileNetV2 (ImageNet pretrained, fine-tuned)
- **Training dataset:** New Plant Diseases Dataset (~87,000 augmented images, 38 classes)
- **Validation accuracy:** 99.00%
- **Weighted F1 score:** 0.9900
- **Macro F1 score:** 0.9901
- **Image size:** 160 × 160
- **Parameters:** ~2.4 M
- **File size:** ~28 MB
- **Classes:** 38 disease types across 14 crops (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)

## Training Details

- **Epochs:** 5 (head) + 15 (fine-tune)
- **Learning rate:** 1e-3 (head), 1e-4 (fine-tune)
- **Batch size:** 64
- **Label smoothing:** 0.1
- **Optimizer:** Adam
- **Augmentation:** Random horizontal flip, rotation, zoom, contrast

## Why this dataset?

The New Plant Diseases Dataset (by [vipoooool](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)) is an augmented version of the original PlantVillage dataset, providing ~87,000 training images vs ~54,000 in the original. The additional augmented samples help the model learn more robust features.

## Setup

### 1. Install Python 3.12

Download from <https://www.python.org/downloads/release/python-3128/> (TensorFlow does not yet support Python 3.13+).

During installation, check **"Add python.exe to PATH"**.

### 2. Create virtual environment and install dependencies

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the trained model

The model file (~28 MB) is hosted on Hugging Face:

**Direct download:**  
https://huggingface.co/ashikaasriarun/MobileNet-V2-NewPlantDisease/resolve/main/best_mnv2_v2.keras

**Model page:**  
https://huggingface.co/ashikaasriarun/MobileNet-V2-NewPlantDisease

Place `best_mnv2_v2.keras` in the same folder as `app.py`.

**Note:** The app will auto-download the model on first run if it's not found locally.

### 4. Run the app

```bash
python app.py
```

Open <http://127.0.0.1:7860> in your browser. Upload a leaf image to get a disease prediction.

## Files

| File | Description |
|---|---|
| `app.py` | Gradio interface for live inference |
| `labels.json` | Class index → name mapping (38 classes) |
| `requirements.txt` | Python dependencies |
| `best_mnv2_v2.keras` | Trained model weights (download from Hugging Face) |

## Comparison with other models

| Model + Dataset | Accuracy | F1 Score | File Size |
|---|---|---|---|
| **MobileNetV2 + New Plant Diseases (this)** | **99.00%** | **0.9900** | ~28 MB |
| MobileNetV2 + Original PlantVillage | 99.36% | 0.9943 | ~28 MB |
| EfficientNet-B0 + Original PlantVillage | 99.71% | 0.9972 | ~50 MB |

## License

MIT — see [LICENSE](LICENSE).

## Citation

Dataset:

> Bhattarai, S. (2018). *New Plant Diseases Dataset (Augmented)*. Kaggle.  
> https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Based on the original PlantVillage dataset:

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
