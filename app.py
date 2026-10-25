import os

# Render CPU performance settings — set before importing torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import uuid
from pathlib import Path

import torch
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify,
    url_for,
)
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================================================
# 1. CPU PERFORMANCE SETTINGS
# =========================================================

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def positive_int_env(name, default, minimum=64):
    """Read a positive integer environment variable safely."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default

    return max(minimum, value)


# Maximum image side used by the neural network.
# Lower values generally mean faster inference on CPU.
INFERENCE_MAX_SIDE = positive_int_env("ADAIN_MAX_SIDE", 320)

# Avoid unnecessarily large output images.
OUTPUT_MAX_SIDE = positive_int_env("ADAIN_MAX_OUTPUT_SIDE", 1600)


# =========================================================
# 2. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"
EXAMPLES_DIR = BASE_DIR / "examples"

VGG_PATH = BASE_DIR / "vgg_normalised.pth"

DECODER_PATH = (
    BASE_DIR
    / "experiment"
    / "alpha_fixed_v1"
    / "decoder_10.pth"
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATE_DIR),
)

app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-development-key-change-before-deployment",
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)


# =========================================================
# 3. DEVICE SELECTION
# =========================================================

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


device = select_device()

print("Using device:", device, flush=True)
print("Inference maximum side:", INFERENCE_MAX_SIDE, flush=True)
print("Output maximum side:", OUTPUT_MAX_SIDE, flush=True)


# =========================================================
# 4. LOAD MODELS
# =========================================================

if not VGG_PATH.is_file():
    raise FileNotFoundError(
        f"VGG checkpoint not found: {VGG_PATH}"
    )

if not DECODER_PATH.is_file():
    raise FileNotFoundError(
        f"Trained decoder not found: {DECODER_PATH}"
    )

print("Loading VGG encoder...", flush=True)

encoder = VGGEncoder(str(VGG_PATH)).to(device)

print("Loading alpha-aware decoder...", flush=True)

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        str(DECODER_PATH),
        map_location=device,
        weights_only=True,
    )
)

encoder.eval()
decoder.eval()

for parameter in encoder.parameters():
    parameter.requires_grad_(False)

for parameter in decoder.parameters():
    parameter.requires_grad_(False)

print("VGG loaded:", VGG_PATH, flush=True)
print("DECODER LOADED:", DECODER_PATH, flush=True)


# =========================================================
# 5. UPLOAD FORM
# =========================================================

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=0.5)
    submit = SubmitField("Transfer Style")


# =========================================================
# 6. FILE AND IMAGE HELPERS
# =========================================================

def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def save_uploaded_file(uploaded_file, prefix):
    """Save an uploaded image using a unique filename."""

    original_name = secure_filename(
        uploaded_file.filename or ""
    )

    if not allowed_file(original_name):
        raise ValueError(
            "Please upload a PNG, JPG, or JPEG image."
        )

    extension = Path(original_name).suffix.lower()

    stored_name = f"{prefix}_{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / stored_name

    uploaded_file.save(str(destination))

    return stored_name


def get_existing_upload(field_name):
    """Safely reuse an image previously selected in the UI."""

    upload_root = UPLOAD_DIR.resolve()

    for value in request.form.getlist(field_name):
        if not value:
            continue

        filename = value.strip()

        # Accept only a filename, not a filesystem path.
        if os.path.basename(filename) != filename:
            continue

        if not allowed_file(filename):
            continue

        candidate = (UPLOAD_DIR / filename).resolve()

        if candidate.parent != upload_root:
            continue

        if candidate.is_file():
            return filename

    return None


def resize_longest_side(image, max_side):
    """Resize proportionally without cropping the image."""

    width, height = image.size
    longest_side = max(width, height)

    if longest_side <= max_side:
        return image

    scale = max_side / float(longest_side)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def save_output_tensor(tensor, path, output_size):
    """Save generated tensor as a PNG at the requested dimensions."""

    image_tensor = (
        tensor.detach()
        .cpu()
        .squeeze(0)
        .clamp(0, 1)
    )

    output_image = transforms.ToPILImage()(image_tensor)

    if output_image.size != output_size:
        output_image = output_image.resize(
            output_size,
            Image.Resampling.LANCZOS,
        )

    output_image.save(str(path), format="PNG")


def is_ajax_request():
    return (
        request.headers.get("X-Requested-With")
        == "XMLHttpRequest"
    )


def error_response(message, status=400):
    if is_ajax_request():
        return jsonify({
            "success": False,
            "error": message,
        }), status

    return render_template(
        "index.html",
        form=UploadForm(),
        result_image=None,
        content_image=None,
        style_image=None,
        error=message,
        alpha=0.5,
    ), status


# =========================================================
# 7. STYLE TRANSFER
# =========================================================

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device,
):
    """
    Apply AdaIN style transfer.

    Alpha 0 = original image.
    Alpha 1 = full style transfer.
    """

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "Alpha must be a number between 0 and 1."
        )

    if alpha == 0.0:
        return (
            transforms.ToTensor()(content_image)
            .unsqueeze(0)
            .to(device)
        )

    # Reduce the inference workload while preserving aspect ratio.
    content_small = resize_longest_side(
        content_image,
        INFERENCE_MAX_SIDE,
    )

    style_small = resize_longest_side(
        style_image,
        INFERENCE_MAX_SIDE,
    )

    to_tensor = transforms.ToTensor()

    content_tensor = (
        to_tensor(content_small)
        .unsqueeze(0)
        .to(device)
    )

    style_tensor = (
        to_tensor(style_small)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():

        content_features = encoder(content_tensor)
        style_features = encoder(style_tensor)

        content_feat = content_features[-1]
        style_feat = style_features[-1]

        target = adaptive_instance_normalization(
            content_feat,
            style_feat,
        )

        # Alpha interpolation
        blended_features = (
            (1.0 - alpha) * content_feat
            + alpha * target
        )

        generated = decoder(blended_features)
        generated = generated.clamp(0, 1)

        # Match the preprocessed content dimensions.
        target_height = content_tensor.shape[-2]
        target_width = content_tensor.shape[-1]

        generated = generated[
            :,
            :,
            :target_height,
            :target_width,
        ]

    return generated


# =========================================================
# 8. MAIN ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None
    alpha = 0.5

    if request.method == "POST":

        # ---------------------------------
        # Read and validate alpha
        # ---------------------------------

        try:
            alpha = float(
                request.form.get("alpha", "0.5")
            )

            if (
                not math.isfinite(alpha)
                or not 0.0 <= alpha <= 1.0
            ):
                raise ValueError(
                    "Alpha must be between 0 and 1."
                )

        except (TypeError, ValueError) as exc:
            return error_response(
                str(exc) or "Invalid alpha value."
            )

        print("=" * 40, flush=True)
        print("Alpha received:", alpha, flush=True)
        print("Active decoder:", DECODER_PATH, flush=True)
        print("=" * 40, flush=True)

        # ---------------------------------
        # Save uploads or reuse selected files
        # ---------------------------------

        try:
            content_file = request.files.get("content")
            style_file = request.files.get("style")

            if content_file and content_file.filename:
                content_filename = save_uploaded_file(
                    content_file,
                    "content",
                )
            else:
                content_filename = get_existing_upload(
                    "content_path"
                )

            if style_file and style_file.filename:
                style_filename = save_uploaded_file(
                    style_file,
                    "style",
                )
            else:
                style_filename = get_existing_upload(
                    "style_path"
                )

        except ValueError as exc:
            return error_response(str(exc))

        if not content_filename or not style_filename:
            return error_response(
                "Please upload both content and style images."
            )

        content_path = UPLOAD_DIR / content_filename
        style_path = UPLOAD_DIR / style_filename

        if (
            not content_path.is_file()
            or not style_path.is_file()
        ):
            return error_response(
                "An uploaded image could not be found. "
                "Please upload it again."
            )

        # ---------------------------------
        # Run style transfer
        # ---------------------------------

        try:
            with Image.open(content_path) as source:
                content_image = source.convert("RGB")

            with Image.open(style_path) as source:
                style_image = source.convert("RGB")

            print("Running style transfer...", flush=True)
            print(
                "Content size:",
                content_image.size,
                flush=True,
            )
            print(
                "Style size:",
                style_image.size,
                flush=True,
            )

            # Unique name: never overwrite previous generated results.
            result_filename = (
                f"stylized_{uuid.uuid4().hex}.png"
            )

            result_path = UPLOAD_DIR / result_filename

            if alpha == 0.0:

                # Alpha 0 returns the original content image.
                content_image.save(
                    str(result_path),
                    format="PNG",
                )

            else:

                # Avoid oversized output images on small servers.
                # Normal-sized images are not enlarged here.
                content_for_output = resize_longest_side(
                    content_image,
                    OUTPUT_MAX_SIDE,
                )

                output_size = content_for_output.size

                generated = style_transfer(
                    content_for_output,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device,
                )

                save_output_tensor(
                    generated,
                    result_path,
                    output_size=output_size,
                )

            result_image = result_filename

            print(
                "Style transfer completed:",
                result_path,
                flush=True,
            )

            # Keep the JSON format expected by index.html.
            if is_ajax_request():
                return jsonify({
                    "success": True,
                    "result_image_url": url_for(
                        "send_image",
                        filename=result_image,
                    ),
                    "content_path": content_filename,
                    "style_path": style_filename,
                })

        except Exception as exc:
            app.logger.exception(
                "Style transfer failed"
            )

            error = f"Style transfer failed: {exc}"

            if is_ajax_request():
                return jsonify({
                    "success": False,
                    "error": error,
                }), 500

    # ---------------------------------
    # Normal HTML response
    # ---------------------------------

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
        alpha=alpha,
    )


# =========================================================
# 9. IMAGE ROUTES
# =========================================================

@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(
        str(UPLOAD_DIR),
        filename,
    )


@app.route("/examples/<path:filename>")
def send_examples(filename):
    return send_from_directory(
        str(EXAMPLES_DIR),
        filename,
    )


# =========================================================
# 10. LOCAL STARTUP
# =========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False,
    )