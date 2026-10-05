# SnapAI — Model Training & Upgrade Guide

## Current Architecture vs Production Architecture

| Component | Current (Ships Now) | Production (6–12 months) |
|---|---|---|
| Moment Detection | VLM visual features + cosine scoring | Fine-tuned CLIP / ViT-B/32 |
| Shot Quality | Laplacian + cascade heuristics | ResNet-18 trained on aesthetic datasets |
| Gaze Detection | Iris centering + symmetry | MPIIGaze / GazeNet |
| Face Detection | OpenCV Haar Cascades | YOLOv8-face or MediaPipe |

---

## Step 1: Upgrade Face Detection (Do This First — Biggest Impact)

### Option A: MediaPipe (easiest, free, runs on CPU)
```bash
pip install mediapipe
```
```python
import mediapipe as mp
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.5)
results  = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

### Option B: YOLOv8-face
```bash
pip install ultralytics
```
```python
from ultralytics import YOLO
model = YOLO("yolov8n-face.pt")
results = model(frame)
```

---

## Step 2: Real CLIP Moment Detection

### Install
```bash
pip install transformers torch torchvision
```

### Replace VLMDetector.detect() with:
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

MOMENT_TEXTS = [
    "a photo of people cutting a wedding cake",
    "a photo of a ring ceremony at a wedding",
    "a photo of people dancing at a wedding",
    "a photo of a group of people posing for a photo",
    "a photo of someone blowing birthday candles",
    "a photo of people celebrating with confetti",
]

def detect_with_clip(frame):
    img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=MOMENT_TEXTS, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
    best_idx = probs.index(max(probs))
    return MOMENT_TEXTS[best_idx], probs[best_idx]
```

**Cost**: ~100ms per frame on CPU, ~8ms on GPU.  
**Cloud alternative**: Use OpenAI Vision API ($0.01/frame) for serverless version.

---

## Step 3: Training Your Own Shot Quality Model

### Dataset you need
- **AVA Dataset**: 250K photos rated 1–10 by humans. Free download.
  https://github.com/mtobeiyf/ava_downloader
- **CUHK-PQ**: Personal photo quality dataset.
- Your own labeled event photos (collect 2,000+ labeled good/bad shots)

### Training pipeline
```python
import torch
import torchvision.models as models
from torch import nn

class ShotQualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier[-1] = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.backbone(x))

# Training loop
model    = ShotQualityNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(20):
    for images, scores in dataloader:
        pred = model(images).squeeze()
        loss = criterion(pred, scores)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(), "shot_quality_net.pth")
```

**Time to train**: 4 hours on a T4 GPU (Google Colab free).

---

## Step 4: Data Collection Strategy

### For moment detection training data:
1. **YouTube**: Download wedding/event videos, extract frames at key moments
2. **Wedding photography blogs**: Scrape high-quality labeled photos
3. **Getty/Shutterstock API**: Use their labeled collections

### Labeling tool:
```bash
pip install labelImg   # GUI labeler
```
Label schema: `{moment_type: str, quality: 0-10, has_face: bool, is_best: bool}`

### Minimum dataset sizes:
| Task | Min Samples | Recommended |
|---|---|---|
| Moment detection | 500/class | 2,000/class |
| Shot quality | 5,000 | 50,000+ |
| Gaze detection | 2,000 | 10,000+ |

---

## Step 5: Cloud Deployment (When Ready for Users)

### Architecture
```
Users → Cloudflare CDN
            ↓
       Load Balancer
            ↓
   ┌────────────────┐
   │  WebSocket     │  ← EC2 c5.xlarge ($0.17/hr) or
   │  Server Pool   │    Lambda + API Gateway
   └────────────────┘
            ↓
     S3 (photo storage, $0.023/GB)
            ↓
     RDS PostgreSQL (session/album metadata)
```

### Cost estimate (1,000 events/month):
- EC2 compute: $120/month
- S3 storage: $20/month  
- Data transfer: $30/month
- **Total: ~$170/month**
- Break-even: 34 events @ ₹5,000/event

---

## Immediate Action Items (In Order)

1. **Week 1**: Install MediaPipe, replace Haar cascade → 3x better face detection
2. **Week 2**: Test on 5 real event videos, tune thresholds
3. **Week 3**: Add OpenAI Vision API as CLIP backend (pay-per-call, no training needed)
4. **Month 2**: Build React Native mobile app (camera access is better than browser)
5. **Month 3**: Fine-tune on your own collected event data
6. **Month 6**: Deploy to cloud, launch beta with 10 photographer partners
