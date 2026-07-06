import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, "/root/autodl-tmp/person_reid_osnet/deep-person-reid")
import torchreid


def parse_pid_cam(path):
    name = Path(path).name
    match = re.match(r"([-\d]+)_c(\d+)", name)
    if not match:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), str(path)


def load_checkpoint(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    fixed = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("classifier."):
            continue
        fixed[k] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    print(f"Loaded weights: {weight_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


@torch.no_grad()
def extract_features(model, paths, batch_size, device):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loader = DataLoader(
        ImagePathDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    features = []
    out_paths = []

    model.eval()
    for imgs, batch_paths in loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        features.append(feats.cpu().numpy())
        out_paths.extend(batch_paths)

    return np.concatenate(features, axis=0), out_paths


def make_contact_sheet(query_path, ranked_paths, ranked_scores, out_path):
    thumb_w, thumb_h = 128, 256
    margin = 20
    label_h = 40
    cols = len(ranked_paths) + 1
    width = cols * thumb_w + (cols + 1) * margin
    height = thumb_h + label_h + margin * 2

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    def paste_img(path, x, title):
        img = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
        canvas.paste(img, (x, margin + label_h))
        draw.text((x, margin), title, fill=(0, 0, 0))

    paste_img(query_path, margin, "Query")

    for i, (path, score) in enumerate(zip(ranked_paths, ranked_scores), start=1):
        x = margin + i * (thumb_w + margin)
        pid, cam = parse_pid_cam(path)
        paste_img(path, x, f"Top{i} id={pid} d={score:.3f}")

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Path to query image")
    parser.add_argument("--gallery-dir", default="/root/autodl-tmp/person_reid_osnet/data/market1501/Market-1501-v15.09.15/bounding_box_test")
    parser.add_argument("--weights", default="/root/autodl-tmp/person_reid_osnet/results/osnet_train_20epoch/model/model.pth.tar-20")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", default="/root/autodl-tmp/person_reid_osnet/results/retrieval_examples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gallery_paths = sorted(Path(args.gallery_dir).glob("*.jpg"))
    if not gallery_paths:
        raise RuntimeError(f"No gallery images found: {args.gallery_dir}")

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=751,
        loss="softmax",
        pretrained=False,
    )
    load_checkpoint(model, args.weights)
    model = model.to(device)

    query_feat, _ = extract_features(model, [args.query], args.batch_size, device)
    gallery_feats, gallery_paths = extract_features(model, gallery_paths, args.batch_size, device)

    dists = np.linalg.norm(gallery_feats - query_feat[0:1], axis=1)

    q_pid, q_cam = parse_pid_cam(args.query)
    ranked = np.argsort(dists)

    valid = []
    for idx in ranked:
        g_pid, g_cam = parse_pid_cam(gallery_paths[idx])
        if g_pid == -1:
            continue
        if g_pid == q_pid and g_cam == q_cam:
            continue
        valid.append(idx)
        if len(valid) >= args.topk:
            break

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Query: {args.query}")
    print(f"Query pid={q_pid}, cam={q_cam}")
    print("Top results:")
    for rank, idx in enumerate(valid, start=1):
        path = gallery_paths[idx]
        pid, cam = parse_pid_cam(path)
        print(f"{rank}: pid={pid}, cam={cam}, dist={dists[idx]:.4f}, path={path}")

    out_img = os.path.join(args.out_dir, "retrieval_result.jpg")
    make_contact_sheet(args.query, [gallery_paths[i] for i in valid], [dists[i] for i in valid], out_img)
    print(f"Saved visualization: {out_img}")


if __name__ == "__main__":
    main()
