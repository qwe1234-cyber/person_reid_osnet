import os
import re
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/root/autodl-tmp/person_reid_osnet/deep-person-reid")
import torchreid


PROJECT_ROOT = Path("/root/autodl-tmp/person_reid_osnet")
GALLERY_DIR = PROJECT_ROOT / "data/market1501/Market-1501-v15.09.15/bounding_box_test"
WEIGHTS = PROJECT_ROOT / "results/osnet_train_60epoch/model/model.pth.tar-60"
TOPK = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_pid_cam(path):
    name = Path(path).name
    match = re.match(r"([-\d]+)_c(\d+)", name)
    if not match:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def load_checkpoint(model, weight_path):
    checkpoint = torch.load(str(weight_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    fixed = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("classifier."):
            continue
        fixed[k] = v
    model.load_state_dict(fixed, strict=False)


@torch.no_grad()
def extract_one(model, image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)
    feat = model(x)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()[0]


@torch.no_grad()
def extract_gallery(model):
    paths = sorted(GALLERY_DIR.glob("*.jpg"))
    feats = []

    model.eval()
    for i in range(0, len(paths), 128):
        batch_paths = paths[i:i + 128]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        x = torch.stack(imgs, dim=0).to(DEVICE)
        f = model(x)
        f = torch.nn.functional.normalize(f, p=2, dim=1)
        feats.append(f.cpu().numpy())

    return paths, np.concatenate(feats, axis=0)


print("Loading OSNet model...")
model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=751,
    loss="softmax",
    pretrained=False,
)
load_checkpoint(model, WEIGHTS)
model = model.to(DEVICE)
model.eval()

print("Extracting gallery features, please wait...")
gallery_paths, gallery_feats = extract_gallery(model)
print(f"Gallery loaded: {len(gallery_paths)} images")


def search(image):
    if image is None:
        return [], "Please upload or select a query image."

    q_feat = extract_one(model, image)
    dists = np.linalg.norm(gallery_feats - q_feat[None, :], axis=1)
    ranked = np.argsort(dists)[:TOPK]

    outputs = []
    lines = []
    for rank, idx in enumerate(ranked, start=1):
        path = gallery_paths[idx]
        pid, cam = parse_pid_cam(path)
        dist = float(dists[idx])
        img = Image.open(path).convert("RGB")
        caption = f"Top {rank} | ID {pid} | Cam {cam} | Dist {dist:.4f}"
        outputs.append((img, caption))
        lines.append(caption)

    return outputs, "\n".join(lines)


demo = gr.Interface(
    fn=search,
    inputs=gr.Image(type="pil", label="Query person image"),
    outputs=[
        gr.Gallery(label="Top-5 retrieval results", columns=5, height=320),
        gr.Textbox(label="Retrieval details", lines=6),
    ],
    title="OSNet Person Re-Identification Demo",
    description="Upload a query person image. The system retrieves the most similar people from Market-1501 gallery.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
