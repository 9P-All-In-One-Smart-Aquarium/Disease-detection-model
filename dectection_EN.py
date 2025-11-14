# ============================================================
# Raspberry Pi: Capture camera image every minute → Run inference → Send to Mobius(oneM2M) via HTTPS
# Using images captured from Raspberry Pi camera
# ============================================================
import os, time, json, subprocess
from datetime import datetime
import requests
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------------------------------------------
# 0) Wi-Fi Auto Connection (Added Code)
# ----------------------------------------------------------
WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PSK  = "YOUR_WIFI_PASSWORD"

def ensure_wifi_connected():
    """
    Automatically connects to the configured Wi-Fi SSID
    if the Raspberry Pi is not currently connected.
    Uses nmcli.
    """
    try:
        # Check current Wi-Fi status
        result = subprocess.check_output(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            stderr=subprocess.DEVNULL
        ).decode()

        if WIFI_SSID in result:
            print(f"[WiFi] Already connected → {WIFI_SSID}")
            return

        print(f"[WiFi] Wi-Fi not connected → Attempting to connect to {WIFI_SSID}…")

        # Attempt connection
        subprocess.run(
            ["nmcli", "dev", "wifi", "connect", WIFI_SSID, "password", WIFI_PSK],
            check=False
        )

        print("[WiFi] Connection request sent")

    except Exception as e:
        print(f"[WiFi-ERR] Error occurred during Wi-Fi auto connection: {e}")

# ----------------------------------------------------------
# A) Path / Environment Configuration
# ----------------------------------------------------------
CKPT_PATH = "/YOUR/PATH/fish_disease_resnet18_best.pt"
INTERVAL_SEC = 10
CAMERA_INDEX = 0   # Default camera: /dev/video0

# ----------------------------------------------------------
# B) Mobius Server Configuration (Pattern preserved)
# ----------------------------------------------------------
ca_cert_path = "/YOUR/PATH/rootCA.pem"   # If exists → verify with CA cert, else verify=False
CSE_BASE_URL  = "https://YOUR_WIFI_IP:443"
CSE_BASE_NAME = "Mobius"
AE_NAME       = "AE-Rapi"
CNT_NAME      = "status"
ORIGIN        = "S-Rapi"

def url_cse(): return f"{CSE_BASE_URL}/{CSE_BASE_NAME}"
def url_ae():  return f"{url_cse()}/{AE_NAME}"
def url_cnt(): return f"{url_ae()}/{CNT_NAME}"

def oneM2M_headers(ty: int | None = None) -> dict:
    ri = f"rqi-{int(datetime.utcnow().timestamp()*1000)}"
    h = {
        "Accept": "application/json",
        "X-M2M-Origin": ORIGIN,
        "X-M2M-RI": ri,
        "X-M2M-RVI": "4",
    }
    if ty is not None:
        h["Content-Type"] = f"application/json; ty={ty}"
    return h

verify_ca = os.path.exists(ca_cert_path)
verify_arg = ca_cert_path if verify_ca else False

def publish_code(val:int):
    try:
        payload = {"m2m:cin": {"con": val}}
        res = requests.post(
            url_cnt(),
            headers=oneM2M_headers(ty=4),
            data=json.dumps(payload),
            verify=verify_arg,
            timeout=(3,5)
        )
        if res.status_code in (201, 200):
            print(f"[PUB] con={val} sent successfully")
        else:
            print(f"[ERR] Transmission failed: {res.status_code} {res.text}")
    except Exception as e:
        print(f"[ERR] Exception in publish: {e}")

# ----------------------------------------------------------
# C) Model Load & Preprocessing
# ----------------------------------------------------------
def load_model(ckpt_path:str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt.get("class_names", ["normal","finrot","whitespot"])
    img_size = ckpt.get("img_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    NAME2CODE = {"normal":0, "finrot":1, "whitespot":2}
    idx2code = [NAME2CODE.get(n, 99) for n in class_names]
    print("[INFO] class_names:", class_names)
    print("[INFO] idx2code   :", idx2code)
    return model, tfm, idx2code

@torch.no_grad()
def predict_image(model, tfm, pil_img: Image.Image):
    x = tfm(pil_img).unsqueeze(0)
    prob = torch.softmax(model(x), dim=1)[0]
    idx = int(prob.argmax().item())
    conf = float(prob[idx].item())
    return idx, conf

# ----------------------------------------------------------
# D) Execution (Camera Capture Based)
# ----------------------------------------------------------
def main():
    # Ensure Wi-Fi connection (added)
    ensure_wifi_connected()

    # Load model
    model, tfm, idx2code = load_model(CKPT_PATH)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera. VideoCapture({CAMERA_INDEX}) failed")

    print(f"[RUN] Starting inference every {INTERVAL_SEC} sec + sending to Mobius (Raspberry Pi)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERR] Failed to read camera frame")
                time.sleep(1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            pred_idx, conf = predict_image(model, tfm, pil_img)
            code = idx2code[pred_idx] if 0 <= pred_idx < len(idx2code) else 99

            print(f"[INF] frame → pred_idx={pred_idx}, code={code}, conf={conf:.2f}")

            if code in (0, 1, 2):
                publish_code(code)
            else:
                print("[SKIP] Undefined class (99)")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopped")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
