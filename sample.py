import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

CONTENT_DIR = "/Users/akshaypal/test2017"
STYLE_DIR = "/Users/akshaypal/Downloads/train_3"

VGG_PATH = "/Users/akshaypal/FIRST_NST.PJ/vgg_normalised.pth"
DECODER_PATH = "/Users/akshaypal/FIRST_NST.PJ/experiment/big_dataset/decoder_10.pth"

OUTPUT_PATH = "/Users/akshaypal/FIRST_NST.PJ/experiment/big_dataset/sample_comparison.png"

NUM_IMAGES = 40
IMAGE_SIZE = 256

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

def get_images(folder):

    files = []

    for file in os.listdir(folder):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(folder, file))

    files.sort()

    return files

content_files = get_images(CONTENT_DIR)
style_files = get_images(STYLE_DIR)

print("Content images found:", len(content_files))
print("Style images found:", len(style_files))


# =========================
# LOAD MODELS
# =========================

print("Loading VGG encoder...")

encoder = VGGEncoder(VGG_PATH).to(device)
encoder.eval()

print("Loading trained decoder...")

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device,
        weights_only=True
    )
)

decoder.eval()

print("Models loaded successfully.")

all_images = []

with torch.no_grad():

    i = 0

    while len(all_images) < NUM_IMAGES * 3:

        if i >= len(content_files) or i >= len(style_files):
            break

        content_path = content_files[i]
        style_path = style_files[i]

        try:

            # -------------------------
            # CONTENT IMAGE
            # -------------------------

            content_image = Image.open(
                content_path
            ).convert("RGB")

            content_tensor = transform(
                content_image
            ).unsqueeze(0).to(device)


            # -------------------------
            # STYLE IMAGE
            # -------------------------

            style_image = Image.open(
                style_path
            ).convert("RGB")

            style_tensor = transform(
                style_image
            ).unsqueeze(0).to(device)


            # -------------------------
            # ENCODER
            # -------------------------

            content_features = encoder(content_tensor)
            style_features = encoder(style_tensor)


            # -------------------------
            # AdaIN
            # -------------------------

            target = adaptive_instance_normalization(
                content_features[-1],
                style_features[-1]
            )


            # -------------------------
            # DECODER
            # -------------------------

            generated = decoder(target)

            generated = generated.clamp(0, 1)


            # -------------------------
            # SAVE:
            # CONTENT → STYLE → OUTPUT
            # -------------------------

            all_images.append(content_tensor.cpu())
            all_images.append(style_tensor.cpu())
            all_images.append(generated.cpu())

            print(
                f"Generated {i + 1}/{NUM_IMAGES}"
            )

            i += 1

        except Exception as e:

            print("Skipping:")
            print(content_path)
            print("Reason:", e)

            i += 1


if len(all_images) > 0:

    images = torch.cat(all_images, dim=0)

    save_image(
        images,
        OUTPUT_PATH,
        nrow=3
    )

    print()
    print("================================")
    print("DONE!")
    print("Generated samples:", len(all_images) // 3)
    print("Saved at:")
    print(OUTPUT_PATH)
    print("================================")

else:
     print("No images generated.")