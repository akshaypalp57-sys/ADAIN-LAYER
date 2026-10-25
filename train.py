import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.models import VGGEncoder, Decoder
from utils.utils import (
    ImageFolderDataset,
    get_transform,
    adaptive_instance_normalization,
    calc_mean_std,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train an alpha-aware AdaIN style-transfer decoder."
    )

    parser.add_argument(
        "--content_dir",
        type=str,
        default=str(PROJECT_ROOT / "content_data"),
    )
    parser.add_argument(
        "--style_dir",
        type=str,
        default=str(PROJECT_ROOT / "style_data"),
    )
    parser.add_argument(
        "--vgg",
        type=str,
        default=str(PROJECT_ROOT / "vgg_normalised.pth"),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment1",
    )

    parser.add_argument("--final_size", type=int, default=256)
    parser.add_argument("--content_size", type=int, default=512)
    parser.add_argument("--style_size", type=int, default=512)

    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-crop to disable random cropping.",
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--content_weight", type=float, default=1.5)
    parser.add_argument("--style_weight", type=float, default=3.0)

    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=2)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load an existing decoder checkpoint.",
    )
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--optimizer_path", type=str, default=None)

    return parser.parse_args()


def load_checkpoint(path, device):
    """Load a trusted local checkpoint onto the selected device."""
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(
            str(path),
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        # Only use checkpoints from sources you trust.
        return torch.load(str(path), map_location=device)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def save_training_checkpoints(
    decoder,
    optimizer,
    scheduler,
    save_dir,
    epoch_number,
):
    """
    Save separate decoder and optimizer files for compatibility
    with the existing application, plus the scheduler state.
    """
    decoder_path = save_dir / f"decoder_{epoch_number}.pth"
    optimizer_path = save_dir / f"optimizer_{epoch_number}.pth"
    scheduler_path = save_dir / f"scheduler_{epoch_number}.pth"

    torch.save(decoder.state_dict(), decoder_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)

    print(f"\nDecoder saved:  {decoder_path}")
    print(f"Optimizer saved: {optimizer_path}")
    print(f"Scheduler saved: {scheduler_path}")


def main():
    args = parse_arguments()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")

    if args.final_size < 1:
        raise ValueError("--final_size must be positive.")

    if args.save_interval < 1 or args.log_interval < 1:
        raise ValueError(
            "--save_interval and --log_interval must be positive."
        )

    if args.content_weight < 0 or args.style_weight < 0:
        raise ValueError("Loss weights cannot be negative.")

    if args.lr <= 0:
        raise ValueError("--lr must be positive.")

    if args.resume and not args.decoder_path:
        raise ValueError(
            "--resume requires --decoder_path."
        )

    if args.optimizer_path and not args.resume:
        raise ValueError(
            "--optimizer_path requires --resume."
        )

    # Prevent paths such as ../../some_folder from escaping experiment/.
    experiment_name = Path(args.experiment)

    if (
        experiment_name.is_absolute()
        or ".." in experiment_name.parts
        or len(experiment_name.parts) != 1
        or args.experiment in ("", ".", "..")
    ):
        raise ValueError(
            "--experiment must be a simple folder name, "
            "such as alpha_fixed_v1."
        )

    save_dir = PROJECT_ROOT / "experiment" / args.experiment

    # IMPORTANT: do not overwrite an existing experiment.
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(
            f"\nExperiment folder already contains files:\n{save_dir}\n"
            "Choose a NEW --experiment name. Existing checkpoints "
            "have not been modified."
        )

    content_dir = Path(args.content_dir).expanduser().resolve()
    style_dir = Path(args.style_dir).expanduser().resolve()
    vgg_path = Path(args.vgg).expanduser().resolve()

    for folder, label in [
        (content_dir, "Content dataset"),
        (style_dir, "Style dataset"),
    ]:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"{label} directory not found: {folder}"
            )

    if not vgg_path.is_file():
        raise FileNotFoundError(
            f"VGG checkpoint not found: {vgg_path}"
        )

    device = select_device()

    print("\n======================================")
    print("ALPHA-AWARE ADAIN TRAINING")
    print("======================================")
    print("Device:", device)
    print("Content directory:", content_dir)
    print("Style directory:", style_dir)
    print("VGG checkpoint:", vgg_path)
    print("Experiment:", args.experiment)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Content weight:", args.content_weight)
    print("Style weight:", args.style_weight)
    print("Resume:", args.resume)
    print("Output directory:", save_dir)
    print("======================================\n")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Record configuration without modifying any previous experiment.
    args.content_dir = str(content_dir)
    args.style_dir = str(style_dir)
    args.vgg = str(vgg_path)

    with open(save_dir / "args.txt", "x", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.final_size,
    )

    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.final_size,
    )

    content_dataset = ImageFolderDataset(
        str(content_dir),
        content_transform,
    )

    style_dataset = ImageFolderDataset(
        str(style_dir),
        style_transform,
    )

    if len(content_dataset) < args.batch_size:
        raise ValueError(
            f"Not enough content images: {len(content_dataset)}"
        )

    if len(style_dataset) < args.batch_size:
        raise ValueError(
            f"Not enough style images: {len(style_dataset)}"
        )

    # MPS/CPU do not need CUDA pinned-memory loading.
    pin_memory = device.type == "cuda"

    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

    num_batches = min(
        len(content_loader),
        len(style_loader),
    )

    if num_batches == 0:
        raise ValueError(
            "No complete training batches. Check dataset sizes "
            "and batch_size."
        )

    print("Content images:", len(content_dataset))
    print("Style images:", len(style_dataset))
    print("Content batches:", len(content_loader))
    print("Style batches:", len(style_loader))
    print("Batches processed per epoch:", num_batches)

    # Load the fixed pretrained encoder and freeze its parameters.
    encoder = VGGEncoder(str(vgg_path)).to(device)
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    decoder = Decoder().to(device)

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr,
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            1.0 / (1.0 + args.lr_decay * epoch)
        ),
    )

    if args.resume:
        decoder_checkpoint = load_checkpoint(
            args.decoder_path,
            device,
        )

        decoder.load_state_dict(decoder_checkpoint)

        print("Loaded decoder:", args.decoder_path)

        if args.optimizer_path:
            optimizer_checkpoint = load_checkpoint(
                args.optimizer_path,
                device,
            )

            optimizer.load_state_dict(optimizer_checkpoint)

            print("Loaded optimizer:", args.optimizer_path)
        else:
            print(
                "No optimizer checkpoint supplied. "
                "Starting with a fresh optimizer."
            )

        print(
            "Scheduler starts a new schedule for this run; "
            "its previous state was not restored."
        )

    decoder.train()
    mse_loss = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        decoder.train()

        total_loss = 0.0
        total_content_loss = 0.0
        total_style_loss = 0.0
        processed_batches = 0

        progress_bar = tqdm(
            zip(content_loader, style_loader),
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # Frozen encoder features need no gradients.
            with torch.no_grad():
                content_features = encoder(content_batch)
                style_features = encoder(style_batch)

                content_feat = content_features[-1]
                style_feat = style_features[-1]

                adain_target = adaptive_instance_normalization(
                    content_feat,
                    style_feat,
                )

            # Each image in the batch receives a random alpha.
            alpha = torch.rand(
                content_batch.size(0),
                1,
                1,
                1,
                device=device,
            )

            # Interpolate content and AdaIN target features.
            interpolated_target = (
                (1.0 - alpha) * content_feat
                + alpha * adain_target
            )

            # Generate the stylized image.
            generated = decoder(interpolated_target)

            # These features must remain differentiable with respect
            # to generated, even though encoder parameters are frozen.
            generated_features = encoder(generated)

            content_loss = (
                mse_loss(
                    generated_features[-1],
                    interpolated_target,
                )
                * args.content_weight
            )

            # Alpha-aware style-statistics target:
            # alpha=0 -> content statistics
            # alpha=1 -> style statistics
            style_loss = torch.zeros(
                (),
                device=device,
            )

            for generated_feat, content_feat_level, style_feat_level in zip(
                generated_features,
                content_features,
                style_features,
            ):
                generated_mean, generated_std = calc_mean_std(
                    generated_feat
                )

                content_mean, content_std = calc_mean_std(
                    content_feat_level
                )

                style_mean, style_std = calc_mean_std(
                    style_feat_level
                )

                target_mean = (
                    (1.0 - alpha) * content_mean
                    + alpha * style_mean
                )

                target_std = (
                    (1.0 - alpha) * content_std
                    + alpha * style_std
                )

                style_loss = style_loss + (
                    mse_loss(generated_mean, target_mean)
                    + mse_loss(generated_std, target_std)
                )

            style_loss = style_loss * args.style_weight
            loss = content_loss + style_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()
            processed_batches += 1

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                content=f"{content_loss.item():.4f}",
                style=f"{style_loss.item():.4f}",
            )

        if processed_batches == 0:
            raise RuntimeError(
                "No batches were processed during this epoch."
            )

        # Divide by the batches actually processed, not by the
        # length of one entire dataset.
        average_loss = total_loss / processed_batches
        average_content_loss = (
            total_content_loss / processed_batches
        )
        average_style_loss = (
            total_style_loss / processed_batches
        )

        scheduler.step()

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} | "
            f"Loss: {average_loss:.6f} | "
            f"Content: {average_content_loss:.6f} | "
            f"Style: {average_style_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

        epoch_number = epoch + 1

        # Save at the requested interval AND at the final epoch.
        should_save = (
            epoch_number % args.save_interval == 0
            or epoch_number == args.epochs
        )

        if should_save:
            save_training_checkpoints(
                decoder,
                optimizer,
                scheduler,
                save_dir,
                epoch_number,
            )

            # Save the latest processed training examples.
            with torch.no_grad():
                preview = torch.cat(
                    [
                        content_batch.detach(),
                        style_batch.detach(),
                        generated.detach().clamp(0, 1),
                    ],
                    dim=0,
                )

                save_image(
                    preview,
                    save_dir / f"output_{epoch_number}.png",
                    nrow=args.batch_size,
                )

            print(
                "Preview saved:",
                save_dir / f"output_{epoch_number}.png",
            )

    print("\n======================================")
    print("TRAINING COMPLETED")
    print("Output folder:", save_dir)
    print("======================================")


if __name__ == "__main__":
    main()