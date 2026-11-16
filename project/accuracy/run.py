"""Check ResNet-18 accuracy with exact, PoT, APoT, or Mitchell arithmetic."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

# Use the arithmetic functions from this folder
from .shiftmac import apot_quantize, pot_quantize
from .mitchell_layers import replace_with_mitchell

FIELDS = ["workload", "pe", "bits", "weights_id", "dataset", "subset_seed",
          "num_images", "num_correct", "top1_pct"]


class LabeledImages(Dataset):
    """ImageFolder with class folders mapped to original ImageNet class numbers."""

    def __init__(self, folder, transform, class_map=None):
        self.data = ImageFolder(folder, transform=transform)
        if class_map is None:
            if len(self.data.classes) != 1000:
                raise ValueError("A subset with fewer than 1000 class folders needs --class-map. "
                                 "Use a JSON mapping folder names to ImageNet class indices (0..999).")
            self.labels = list(range(1000))
        else:
            with open(class_map, encoding="utf-8") as f:
                mapping = json.load(f)
            missing = [name for name in self.data.classes if name not in mapping]
            if missing:
                raise ValueError("Class map is missing: " + ", ".join(missing[:5]))
            self.labels = [int(mapping[name]) for name in self.data.classes]
            if any(label < 0 or label >= 1000 for label in self.labels):
                raise ValueError("All ImageNet class indices must be between 0 and 999")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, class_in_folder = self.data[index]
        return img, self.labels[class_in_folder]


def apply_weight_approx(model, pe, bits):
    """Change only Conv2d and Linear weights, leaving the original model untouched."""
    quantize = pot_quantize if pe == "pot" else apot_quantize
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                values = layer.weight.detach().cpu().numpy()
                approx = quantize(values, bits)
                layer.weight.copy_(torch.from_numpy(approx).to(layer.weight))


def evaluate(model, loader, device):
    """Count how many predictions match the image labels."""
    correct = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            count += labels.numel()
    return int(correct), int(count)


def choose_indices(count, limit, seed):
    """Choose the same image indices for all PE types."""
    if limit <= 0 or limit >= count:
        return list(range(count))
    return sorted(random.Random(seed).sample(range(count), limit))


def save_results(filename, rows, append=False):
    """Write one CSV with the accuracy numbers."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    old_rows = []
    if append and filename.exists():
        with filename.open(newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(old_rows + rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ResNet-18 top-1 accuracy")
    parser.add_argument("--data", required=True, help="ImageNet validation folder with class subfolders")
    parser.add_argument("--class-map", help="JSON mapping class folder names to ImageNet class indices")
    parser.add_argument("--pes", nargs="+", choices=["baseline", "pot", "apot", "mitchell"],
                        default=["baseline", "pot", "apot"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", default="project/results/accuracy.csv")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--download-weights", action="store_true",
                        help="Allow torchvision to download pretrained model weights")
    return parser.parse_args()


def check_cached_weights(weights, allow_download):
    """Avoid a surprise model download on a nearly full drive."""
    if allow_download:
        return
    filename = weights.url.rsplit("/", 1)[-1]
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.exists():
        raise FileNotFoundError(f"Pretrained weights are not cached: {path}\n"
                                "Use --download-weights to allow downloading them. "
                                "Set TORCH_HOME to D:/torch-cache to keep them on D:.")


def main():
    args = parse_args()
    if args.bits < 2 or args.batch_size < 1 or args.max_images < 0:
        raise ValueError("bits >= 2, batch-size >= 1 and max-images >= 0 are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if "mitchell" in args.pes:
        print("Mitchell replaces actual products and may be extremely slow. Start with --max-images 1.")
    weights = ResNet18_Weights.IMAGENET1K_V1
    check_cached_weights(weights, args.download_weights)
    images = LabeledImages(args.data, weights.transforms(), args.class_map)
    if len(images) == 0:
        raise ValueError("The dataset contains no images")
    indices = choose_indices(len(images), args.max_images, args.seed)
    loader = DataLoader(Subset(images, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=0)  # Windows-friendly
    rows = []
    for pe in args.pes:
        print(f"Running {pe} on {len(indices)} images", flush=True)
        model = resnet18(weights=weights)
        if pe in ("pot", "apot"):
            apply_weight_approx(model, pe, args.bits)
        elif pe == "mitchell":
            replace_with_mitchell(model)
        model = model.to(args.device)
        correct, total = evaluate(model, loader, args.device)
        row = {"workload": "resnet18", "pe": pe, "bits": args.bits,
               "weights_id": "ResNet18_Weights.IMAGENET1K_V1",
               "dataset": str(Path(args.data).resolve()), "subset_seed": args.seed,
               "num_images": total, "num_correct": correct,
               "top1_pct": round(100.0 * correct / total, 4)}
        rows.append(row)
        print(f"{pe}: {correct}/{total} correct ({row['top1_pct']:.2f}%)", flush=True)
    save_results(args.out, rows, args.append)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
