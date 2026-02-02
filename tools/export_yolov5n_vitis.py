#!/usr/bin/env python3
"""
Export YOLOv5n weights in Vitis/embedded format (BN separate).
Output: weights_vitis.bin, weights_map_vitis.json, model_meta_vitis.json
Format must match: model.X.bn.bias, bn.running_mean, bn.running_var, bn.weight, conv.weight per block.
"""

import sys
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict

YOLOV5_ROOT = Path(__file__).parent.parent / "third_party" / "yolov5"
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

try:
    from models.experimental import attempt_load
except ImportError as e:
    print(f"Error: {e}. Run: pip install -r third_party/yolov5/requirements.txt")
    sys.exit(1)


def load_format_map(format_map_path):
    """Load reference weights_map.json and return ordered list of (key, expected_shape)."""
    with open(format_map_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Allow trailing whitespace / one extra char (e.g. BOM or duplicate brace)
    raw = raw.strip()
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(raw)
    return [(k, tuple(v["shape"])) for k, v in data.items()]


def export_yolov5n_vitis(model_pt_path, output_dir, format_map_path, suffix="vitis"):
    """
    Export YOLOv5n to Vitis format: BN separate, key order from format_map_path.
    suffix: e.g. "vitis" -> weights_vitis.bin; "vitis_ok" -> weights_vitis_ok.bin
    Math: y = W·x → BN(y) → SiLU(y), same result as fused W_fused·x + b_fused → SiLU(y).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load format map for key order and expected shapes
    print(f"Using format map: {format_map_path}")
    key_order = load_format_map(format_map_path)
    print(f"Format has {len(key_order)} tensors.")

    # 2) Load model WITHOUT fuse so we have BN params in state_dict
    print(f"Loading model: {model_pt_path}")
    model = attempt_load(model_pt_path, device="cpu", inplace=False, fuse=False)
    state = model.state_dict()

    # 3) Build bin and new map in format order
    bin_chunks = []
    new_map = OrderedDict()
    offset = 0
    missing = []
    shape_mismatch = []

    for key, expected_shape in key_order:
        if key not in state:
            missing.append(key)
            continue
        t = state[key]  # noqa: state may have extra keys; we only write keys in key_order
        arr = t.detach().cpu().numpy().astype(np.float32)
        actual_shape = tuple(arr.shape)
        if actual_shape != expected_shape:
            shape_mismatch.append((key, expected_shape, actual_shape))
        numel = arr.size
        chunk = arr.tobytes()
        bin_chunks.append(chunk)
        new_map[key] = {
            "offset": offset,
            "shape": list(actual_shape),
            "dtype": "float32",
            "numel": int(numel),
        }
        offset += len(chunk)

    if missing:
        print(f"Error: format map has {len(missing)} keys not in model: {missing[:10]}...")
        sys.exit(1)
    if shape_mismatch:
        for k, exp, act in shape_mismatch[:5]:
            print(f"Shape mismatch {k}: expected {exp}, got {act}")
        if len(shape_mismatch) > 5:
            print(f"... and {len(shape_mismatch) - 5} more")

    weights_binary = b"".join(bin_chunks)
    total_size = len(weights_binary)
    num_params = len(new_map)

    # 4) Save weights_<suffix>.bin
    weights_path = output_dir / f"weights_{suffix}.bin"
    with open(weights_path, "wb") as f:
        f.write(weights_binary)
    print(f"Saved {weights_path} ({total_size} bytes)")

    # 5) Save weights_map_<suffix>.json
    map_path = output_dir / f"weights_map_{suffix}.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(new_map, f, indent=2)
    print(f"Saved {map_path}")

    # 6) model_meta_<suffix>.json — same structure as Vitis model_meta.json
    model_meta = {
        "depth_multiple": 0.33,
        "width_multiple": 0.25,
        "num_classes": 80,
        "input_size": 640,
        "anchors": {
            "p3": [10, 13, 16, 30, 33, 23],
            "p4": [30, 61, 62, 45, 59, 119],
            "p5": [116, 90, 156, 198, 373, 326],
        },
        "channels": {
            "base": [64, 128, 256, 512, 1024],
            "actual": [16, 32, 64, 128, 256],
        },
        "total_weights_size": total_size,
        "num_parameters": num_params,
    }
    meta_path = output_dir / f"model_meta_{suffix}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)
    print(f"Saved {meta_path}")

    return weights_path, map_path, meta_path


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv5n weights in Vitis/embedded format (BN separate, math = fused)"
    )
    parser.add_argument(
        "model",
        type=str,
        default="yolov5n.pt",
        nargs="?",
        help="Path to yolov5n.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: weights/yolov5n)",
    )
    parser.add_argument(
        "--format-map",
        type=str,
        required=True,
        help="Path to reference weights_map.json (Vitis/embedded format for key order)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="unfused",
        help="Output file suffix (default: unfused). e.g. unfused -> weights_unfused.bin",
    )
    args = parser.parse_args()

    out = args.output or str(Path(__file__).parent.parent / "weights" / "yolov5n")
    export_yolov5n_vitis(args.model, out, args.format_map, args.suffix)


if __name__ == "__main__":
    main()
