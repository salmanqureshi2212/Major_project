import os
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import uuid
import uvicorn
# -------- CONFIG --------
MODEL_PATH = "model.h5"
INPUT_DIR = "./inputs"
IMAGE_SIZE_BRAIN = 299   # ⚠️ change based on your model
IMAGE_SIZE_HEART = 256   # ⚠️ change based on your model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BRAIN_MODEL_PATH = os.path.join(BASE_DIR, ".", "models", "brain_model.h5")
HEART_MODEL_PATH = os.path.join(BASE_DIR, ".", "models", "heart_model.pth")
INPUT_DIR= os.path.join(BASE_DIR, ".", "inputs")
OUTPUT_DIR= os.path.join(BASE_DIR, ".", "outputs")
# -------- LOAD MODEL --------
brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH)
print("✅ brain Model loaded")
# -------- LOAD HEART MODEL --------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature (gating signal)
        x: encoder skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class AttentionUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out(d1)

heart_model = AttentionUNet(num_classes=3)  # adjust classes
checkpoint = torch.load(HEART_MODEL_PATH, map_location=DEVICE)

heart_model.load_state_dict(checkpoint["model_state_dict"])
heart_model.to(DEVICE)
heart_model.eval()

print("✅ heart Model loaded")

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE_HEART, IMAGE_SIZE_HEART)),
    transforms.ToTensor(),
])

# -------- BRAIN --------
# -------- CLASS LABELS --------
class_names = ["Cerebellah-hypoplasia", "encephalocele","mild-ventriculomegaly","moderate-ventriculomegaly","normal"]  # ⚠️ change according to your model

# -------- PREPROCESS --------
def preprocess_brain(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"❌ Cannot read {image_path}")

    image = cv2.resize(image, (IMAGE_SIZE_BRAIN, IMAGE_SIZE_BRAIN))
    image = image / 255.0

    image = np.expand_dims(image, axis=0)  # (1, H, W, 3)

    return image

# -------- PREDICT --------
def predict_brain(image_path):
    img = preprocess_brain(image_path)

    preds = brain_model.predict(img)

    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = preds[0][predicted_class]

    return class_names[predicted_class], confidence

# -------- HEART --------
def predict_heart(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"❌ Failed to load {image_path}")

    orig_h, orig_w = image.shape

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = heart_model(img_tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    preds = cv2.resize(preds.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return preds
# -------- COLOR MASK --------
def color_mask(mask):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # class mapping
    colored[mask == 1] = [255, 0, 0]   # cardiac (red)
    colored[mask == 2] = [0, 255, 0]   # thorax (green)

    return colored

def max_horizontal_diameter(mask):
    """
    mask: binary mask (H, W)
    returns: max horizontal diameter in pixels
    """
    rows = np.any(mask, axis=1)
    if not rows.any():
        return 0.0

    diameters = []
    for row in range(mask.shape[0]):
        cols = np.where(mask[row] > 0)[0]
        if len(cols) > 0:
            diameters.append(cols[-1] - cols[0])

    return float(max(diameters)) if diameters else 0.0
# -------- FASTAPI --------
app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/inputs", StaticFiles(directory="inputs"), name="inputs")
@app.post("/api/brain_abnormalities")
async def brain_abnormalities(file: UploadFile = File(...)):
    # -------- GENERATE UNIQUE FILENAME --------
    file_ext = file.filename.split(".")[-1]
    unique_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(INPUT_DIR, unique_name)

    # -------- SAVE FILE --------
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # -------- PREDICT --------
    class_name ,confidence = predict_brain(file_path)
    confidence = float(confidence)
    return {
        "class_name":class_name,
        "confidence": confidence,
        "file_saved_as": unique_name,
        "output_url": f"http://127.0.0.1:8000/outputs/{unique_name}"
}
@app.post("/api/heart_abnormalities")
async def heart_abnormalities(file: UploadFile = File(...)):

    file_ext = file.filename.split(".")[-1]
    unique_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(INPUT_DIR, unique_name)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # 🔥 FIXED
    mask = predict_heart(file_path)
    cardiac_mask = (mask == 1).astype(np.uint8)
    thorax_mask  = (mask == 2).astype(np.uint8)

    cardiac_d = max_horizontal_diameter(cardiac_mask)
    thorax_d  = max_horizontal_diameter(thorax_mask)

    if thorax_d == 0:
        return None

    ctr = cardiac_d / thorax_d
    colored = color_mask(mask)
    out_path = os.path.join(OUTPUT_DIR, unique_name)
    cv2.imwrite(out_path, colored)

    return {
        "class_name": "normal" if ctr<0.55 else "abnormal" ,
        "file_saved_as": unique_name,
        "output_url": f"http://127.0.0.1:8000/outputs/{unique_name}"
}
    
uvicorn.run(app , host= "0.0.0.0", port=8000)