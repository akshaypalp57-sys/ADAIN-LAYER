from pathlib import Path
from datetime import datetime
import hashlib
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

ROOT = Path("/Users/akshaypal/FIRST_NST.PJ")
OLD = Path("/Users/akshaypal/NST_CODE")

CONTENT_DIR = Path("/Users/akshaypal/test2017")
STYLE_DIR = Path("/Users/akshaypal/Downloads/train_3")

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

CANDIDATES = [
    (
        "FIRST_current10",
        ROOT / "experiment/big_dataset/decoder_10.pth",
        ROOT / "vgg_normalised.pth",
    ),
    (
        "NST_v3_10",
        OLD / "experiment/big_dataset_v3/decoder_10.pth",
        OLD / "vgg_normalised.pth",
    ),
    (
        "NST_trial2_10",
        OLD / "experiment/trial2/decoder_10.pth",
        OLD / "vgg_normalised.pth",
    ),
    (
        "NST_trial_4",
        OLD / "experiment/trial/decoder_4.pth",
        OLD / "vgg_normalised.pth",
    ),
]

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

def image_files(folder):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]

content_files = image_files(CONTENT_DIR)
style_files = image_files(STYLE_DIR)

if not content_files or not style_files:
    raise SystemExit(
        "Content/style images nahi mile. CONTENT_DIR aur STYLE_DIR check karo."
    )

# Deterministic test pair: same input for every checkpoint.
content_path = content_files[0]
style_path = style_files[0]

print("Device:", device)
print("Content image:", content_path)
print("Style image:", style_path)

# Same preprocessing as the current app.
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

content_image = Image.open(content_path).convert("RGB")
style_image = Image.open(style_path).convert("RGB")

content = transform(content_image).unsqueeze(0).to(device)
style = transform(style_image).unsqueeze(0).to(device)

content_tile = transforms.ToPILImage()(content[0].cpu())
style_tile = transforms.ToPILImage()(style[0].cpu())

output_dir = ROOT / "experiment" / (
    "checkpoint_alpha_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
output_dir.mkdir(parents=True, exist_ok=False)

column_names = [
    "CONTENT", "STYLE", "alpha=0", "alpha=0.25",
    "alpha=0.5", "alpha=0.75", "alpha=1"
]

successful_rows = []
metrics = []

for name, decoder_path, vgg_path in CANDIDATES:
    if not decoder_path.is_file() or not vgg_path.is_file():
        print("SKIP - missing checkpoint/VGG:", name)
        continue

    print("\nLoading:", name)
    print("Decoder SHA256:", sha256(decoder_path))
    print("VGG SHA256:", sha256(vgg_path))

    try:
        encoder = VGGEncoder(str(vgg_path)).to(device).eval()
        decoder = Decoder().to(device)

        decoder.load_state_dict(
            torch.load(
                str(decoder_path),
                map_location=device,
                weights_only=True,
            )
        )
        decoder.eval()

        row_tiles = [content_tile, style_tile]
        outputs = []

        with torch.inference_mode():
            c_features = encoder(content)
            s_features = encoder(style)

            c_feat = c_features[-1]
            s_feat = s_features[-1]

            target = adaptive_instance_normalization(c_feat, s_feat)

            for alpha in ALPHAS:
                mixed = (1 - alpha) * c_feat + alpha * target
                output = decoder(mixed).clamp(0, 1)[0].cpu()

                outputs.append(output)
                row_tiles.append(transforms.ToPILImage()(output))

        for i in range(1, len(ALPHAS)):
            difference = (
                outputs[i] - outputs[i - 1]
            ).abs().mean().item()

            metrics.append(
                f"{name}: alpha {ALPHAS[i-1]} -> {ALPHAS[i]}, "
                f"mean_pixel_difference={difference:.6f}"
            )

        successful_rows.append((name, row_tiles))
        print("OK:", name)

        del encoder, decoder, c_features, s_features
        del c_feat, s_feat, target, outputs

        if device.type == "mps":
            torch.mps.empty_cache()

    except Exception as e:
        print("FAILED:", name, repr(e))

if not successful_rows:
    raise SystemExit("Koi checkpoint successfully load nahi hua.")

# Build a single comparison image. No existing output is overwritten.
tile_size = 256
label_width = 190
header_height = 52
row_height = 286

sheet = Image.new(
    "RGB",
    (label_width + len(column_names) * tile_size,
     header_height + len(successful_rows) * row_height),
    "#202938",
)

draw = ImageDraw.Draw(sheet)
font = ImageFont.load_default()

for col, label in enumerate(column_names):
    x = label_width + col * tile_size
    draw.text((x + 8, 16), label, font=font, fill="white")

for idx, (name, tiles) in enumerate(successful_rows):
    y = header_height + idx * row_height
    draw.text(
        (10, y + 10),
        name,
        font=font,
        fill="white",
    )

    for col, tile in enumerate(tiles):
        x = label_width + col * tile_size
        sheet.paste(tile.convert("RGB").resize((tile_size, tile_size)),
                    (x, y))

sheet_path = output_dir / "checkpoint_alpha_comparison.png"
sheet.save(sheet_path)

metrics_path = output_dir / "alpha_metrics.txt"
metrics_path.write_text(
    "\n".join(metrics) + "\n",
    encoding="utf-8",
)

print("\n==============================")
print("DIAGNOSTIC COMPLETE")
print("Comparison image:", sheet_path)
print("Alpha metrics:", metrics_path)
print("Output folder:", output_dir)
print("==============================")
