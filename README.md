# OCR Pipeline for Scene Text Detection & Recognition

> **End-to-end OCR system**: combines text detection (YOLOv11) with text recognition (CRNN + CTC) for reading text in real-world images.

---

## Purpose

- **Detect** text regions in natural scenes using YOLOv11.
- **Recognize** cropped text lines using a custom CRNN backbone with CTC decoding.
- Support standard datasets: ICDAR2003, ICDAR2013, ICDAR2015, or custom data.
- Modular design: easy to swap in new detection or recognition backbones.
- Visualize detections and recognized text end-to-end.

---

```
## Project Structure
ocr_project/
├── dataset/ # Raw images, labels for detection, cropped words for recognition
├── models/ # Saved weights for detection & recognition
├── notebook_files # Jupyter/Colab notebooks for training, testing and predictions
├── scripts/ # Scripts for training and evaluation
├── ocr_pipeline.py # Main class for detection + recognition pipeline
├── README.md
```

---

## Train detection
```
Notebook: ocr_text_detection.ipynb
```

## Train recognition
```
Notebook: scripts/ocr_text_recognition.ipynb
```

## Run end-to-end OCR
```
from ocr_pipeline import OCRPipeline

ocr = OCRPipeline(
  text_det_model=your_detection_model,
  text_reg_model=your_crnn_model,
  data_transforms=your_transform,
  idx_to_char=idx_to_char,
  device="cuda"
)

ocr.predict(<image_url>)
```