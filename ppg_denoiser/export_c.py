from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


LAYER_ORDER = ("enc_conv1", "enc_conv2", "dec_conv1", "dec_conv2")


def _format_c_float_array(name: str, arr: np.ndarray) -> str:
    flat = arr.astype(np.float32).reshape(-1)
    body = ", ".join(f"{x:.8e}f" for x in flat)
    return f"static const float {name}[{len(flat)}] = {{{body}}};\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export trained model weights to C header.")
    p.add_argument("--model", type=Path, required=True, help="Trained .keras model path.")
    p.add_argument("--normalization", type=Path, required=True, help="normalization.npz from training.")
    p.add_argument(
        "--output-header",
        type=Path,
        default=Path("edge_c/ppg_denoise_weights.h"),
        help="Generated C header path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model, compile=False)
    norm = np.load(args.normalization)
    feature_mean = norm["feature_mean"].astype(np.float32)
    feature_std = norm["feature_std"].astype(np.float32)
    target_mean = float(norm["target_mean"][0])
    target_std = float(norm["target_std"][0])
    motion_clean_threshold = float(norm["motion_clean_threshold"][0])
    window_len = int(norm["window_len"][0])

    layer_defs = []
    c_arrays = []
    for layer_name in LAYER_ORDER:
        layer = model.get_layer(layer_name)
        kernel, bias = layer.get_weights()
        k, in_ch, out_ch = kernel.shape
        c_arrays.append(_format_c_float_array(f"{layer_name}_kernel", kernel))
        c_arrays.append(_format_c_float_array(f"{layer_name}_bias", bias))
        layer_defs.append((layer_name, int(k), int(in_ch), int(out_ch)))

    args.output_header.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("#ifndef PPG_DENOISE_WEIGHTS_H")
    lines.append("#define PPG_DENOISE_WEIGHTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define PPG_WINDOW_LEN ({window_len})")
    lines.append("#define PPG_INPUT_CHANNELS (7)")
    for name, k, in_ch, out_ch in layer_defs:
        macro = name.upper()
        lines.append(f"#define {macro}_K ({k})")
        lines.append(f"#define {macro}_IN ({in_ch})")
        lines.append(f"#define {macro}_OUT ({out_ch})")
    lines.append("")
    lines.append(f"static const float ppg_target_mean = {target_mean:.8e}f;")
    lines.append(f"static const float ppg_target_std = {target_std:.8e}f;")
    lines.append(f"static const float ppg_motion_clean_threshold = {motion_clean_threshold:.8e}f;")
    lines.append("static const float ppg_adapt_lr = 0.00500000f;")
    lines.append("")
    lines.append(_format_c_float_array("ppg_feature_mean", feature_mean))
    lines.append(_format_c_float_array("ppg_feature_std", feature_std))
    lines.extend(c_arrays)
    lines.append("#endif  // PPG_DENOISE_WEIGHTS_H")
    lines.append("")
    args.output_header.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_header}")


if __name__ == "__main__":
    main()

