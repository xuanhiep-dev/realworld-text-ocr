![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?logo=pytorch)
![YOLOv11](https://img.shields.io/badge/Detection-YOLOv11-green)
![CRNN](https://img.shields.io/badge/Recognition-CRNN-blue)
![CTC](https://img.shields.io/badge/Loss-CTC-yellow)

# OCR System for Detecting and Recognizing Text in Real-World Images

> **End-to-end OCR system**: combines text detection (YOLOv11) with text recognition (CRNN + CTC) for reading text in real-world images.

---

## Purpose

- **Detect** text regions in natural scenes using YOLOv11.
- **Recognize** cropped text lines using a custom CRNN backbone with CTC decoding.
- Support standard datasets: ICDAR2003, ICDAR2013, ICDAR2015, or custom data.
- Modular design: easy to swap in new detection or recognition backbones.
- Visualize detections and recognized text end-to-end.

---

## Key Features
- Text Detection: YOLOv11 for accurate bounding box detection in natural scenes.
- Text Recognition: Custom CRNN backbone with CTC decoding for sequence transcription.
- Dataset Support: ICDAR2003, ICDAR2013, ICDAR2015, and custom datasets.
- Modular Design: Easily swap detection/recognition backbones.
- Visualization: End-to-end detection & recognition results.

---

## Project Structure
```
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

---

## Example Output
👤 **Input Image:**  
*(Street scene with multiple signboards)*  

🤖 **OCR System Output:**  
- 🟩 **Detected Regions**: 5 bounding boxes  
- 🔠 **Recognized Text**:  
  **COFFEE HOUSE** - 0.95   
  **SALE 50%** - 0.92  
  **OPEN** - 0.89  
  **TAXI** - 0.87  
  **BUSES** - 0.84

---

## Roadmap
- Integrate Transformer-based recognizer (e.g., TrOCR).
- Support multilingual datasets (VN/EN/JP).
- Export pipeline to ONNX/TensorRT for real-time inference.
- Web demo with Streamlit or Gradio.

## License
License © 2025 [Duong Xuan Hiep]