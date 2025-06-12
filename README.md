# Player_Tracking_And_Re-Identification
This project detects and tracks players and the ball in a short football video using a fine-tuned YOLOv11 model from Ultralytics and a custom Centroid Tracker.

---

## 📦 Setup Instructions

### 1. Clone YOLOv5 Repository (for compatibility with YOLOv11 weights)

```bash
git clone https://github.com/Jayanth026/Player_Tracking_And_Re-Identification.git
cd Player_Tracking_And_Re-Identification
```

### 2. Install Dependencies

Make sure Python 3.8+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
pip install opencv-python numpy
```

### 3. Place Files

Ensure the following files are in your working directory:

- `best.pt` – Your trained YOLOv11 weights
- `15sec_input_720p.mp4` – Input video file
- `Player_Mapping` – The Python script containing tracking logic
---

## 🚀 How to Run

From your terminal, run the tracking script:

```bash
python Player_Mapping.py
```

This will:

- Load the YOLOv11 model
- Detect players and the ball in each frame
- Assign and track consistent IDs using a Centroid Tracker
- Save the output video as `tracked_output.mp4`

---

## 📚 Dependencies

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- PyTorch
- YOLOv5 (used to load YOLOv11 weights)
- `best.pt` – fine-tuned YOLOv11 weights trained on player + ball detection

---

## 📂 Output

- `tracked_output.mp4` — Output video with bounding boxes and player IDs
- `player_tracking_report.pdf` — Summary report of methodology and challenges

---

## ✅ Notes

- The model used is a basic fine-tuned version of **YOLOv11 by Ultralytics**, trained specifically to detect players and the ball.
- The Centroid Tracker maintains consistent IDs even when players leave and re-enter the frame.
