"""One-off: build static/metrics.json for the /metrics page.

Training curves come from the notebook outputs; the confusion matrix and
per-class scores come from running the deployed model on the test set.
Run:  python make_metrics.py   (needs archive/Birds_25/test)
"""
import json
import re

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from the_app import model, transform, device, bird_info

RUNS = [
    ("ResNet18 · Adam", "resnet18.ipynb", 0.9443, 0.2075),
    ("MobileNetV3 · RMSprop", "mobilnetv3_Rmsprop.ipynb", 0.9568, 0.1443),
    ("MobileNetV3 · SGD", "mobilrnetv3.ipynb", 0.9872, 0.0476),
    ("MobileNetV3 · SGD (deployed)", "mobilnetveepochbetteraccuecy.ipynb", 0.9865, 0.0490),
]


def curves(path):
    nb = json.load(open(path, encoding="utf-8"))
    text = "".join("".join(o.get("text", [])) for c in nb["cells"] for o in c.get("outputs", []))
    train = re.findall(r"Train Loss: ([\d.]+) Acc: ([\d.]+)", text)
    val = re.findall(r"Val Loss: ([\d.]+) Acc: ([\d.]+)", text)
    return {
        "train_loss": [float(l) for l, _ in train], "train_acc": [float(a) for _, a in train],
        "val_loss": [float(l) for l, _ in val], "val_acc": [float(a) for _, a in val],
    }


def evaluate():
    ds = ImageFolder("archive/Birds_25/test", transform=transform)
    assert ds.classes == list(bird_info), "dataset folders must match bird_info order"
    n = len(ds.classes)
    cm = torch.zeros(n, n, dtype=torch.long)
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64):
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, pred):
                cm[t, p] += 1
    tp = cm.diag().float()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    return {
        "test_acc": round((tp.sum() / cm.sum()).item(), 4),
        "confusion": cm.tolist(),
        "per_class": [{"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
                      for p, r, f in zip(precision.tolist(), recall.tolist(), f1.tolist())],
    }


if __name__ == "__main__":
    out = {
        "classes": [v["name"] for v in bird_info.values()],
        "runs": [{"name": name, "test_acc": acc, "test_loss": loss, **curves(nb)} for name, nb, acc, loss in RUNS],
        "deployed": evaluate(),
    }
    json.dump(out, open("static/metrics.json", "w"))
    print("test acc", out["deployed"]["test_acc"])
