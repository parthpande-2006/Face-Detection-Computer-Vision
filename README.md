
# Real-Time Face Detection with a Custom CNN (VGG16 Backbone)

A face detector built from scratch  own dataset, own annotations, own augmentation pipeline, and a custom-trained dual-head neural network that both classifies "face present" and regresses the bounding box, running live on webcam feed. Demo is added.

## What this actually does
Most "face detection" tutorials just call a pretrained model like Haar Cascades or MTCNN. This project trains its own detector end-to-end:
1. **Collected a custom dataset** captured images directly from a live webcam feed (via browser JS inside Colab), then hand-annotated bounding boxes for each face.
2. **Built a bbox-aware augmentation pipeline** with Albumentations — random crops, horizontal/vertical flips, brightness/contrast/gamma shifts, RGB shifts — all while keeping bounding box coordinates correctly transformed alongside the image.
3. **Trained a dual-head CNN** on top of a VGG16 backbone:
   - One head predicts whether a face is present (binary classification)
   - The other head regresses the four bounding box coordinates
4. **Wrote a custom loss function** for the localization task — combining coordinate distance and box-size error — and a custom Keras training loop (`train_step`) to jointly optimize both heads.
5. **Ran real-time inference** by streaming the live webcam feed back through the trained model in-browser, drawing the predicted bounding box on top of the video in real time.

## Pipeline
Webcam capture → Manual annotation (bbox JSON) → Train/val/test split
→ Albumentations augmentation (bbox-consistent)
→ Resize to 120×120, normalize
→ VGG16 backbone → [classification head, regression head]
→ Custom joint loss (BCE + localization loss)
→ Trained FaceTracker model → Real-time webcam inference


## Model architecture

- **Backbone:** VGG16 (ImageNet weights, no top layers)
- **Classification head:** GlobalMaxPooling → Dense(2048, ReLU) → Dense(1, sigmoid)
- **Regression head:** GlobalMaxPooling → Dense(2048, ReLU) → Dense(4, sigmoid)
- **Loss:** Binary cross-entropy (classification) + custom localization loss (regression), combined as `localization_loss + 0.5 * classification_loss`
- **Optimizer:** Adam with a decayed learning rate schedule
- Trained for 10 epochs with TensorBoard logging for loss curves

## Tech stack

- TensorFlow / Keras
- OpenCV
- Albumentations
- NumPy, Matplotlib, Pandas
- Google Colab (webcam capture + GPU training via browser JS integration)

## Results

- Trained on a custom-collected, self-annotated dataset
- Loss curves (classification loss, localization loss, total loss) tracked across train/val splits via TensorBoard
- Real-time bounding box inference tested live via webcam feed

*(Add your actual final loss numbers / a screenshot of the TensorBoard curves here if you have them — makes this section much stronger.)*


## Limitations & future work

- Single-class detector (face only) trained on a small, self-collected dataset — accuracy is bounded by dataset size and diversity
- Single-face assumption per frame (no multi-face NMS/anchor-based detection)
- Could be extended with: a larger/more diverse dataset, multi-face support, a lighter backbone (MobileNet) for faster inference, or ONNX export for deployment outside Colab

## Why this project

Built to understand object detection fundamentals from the ground up — not just calling `cv2.CascadeClassifier()`, but building the annotation-to-inference pipeline myself: data collection, augmentation math, a custom multi-output architecture, and a custom training loop.
