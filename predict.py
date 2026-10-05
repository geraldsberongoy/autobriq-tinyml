"""
Run the int8 model on one or more images exactly the way the ESP32-S3 will:
grayscale -> 96x96 -> int8 (pixel - 128) -> model -> label + confidence.

Usage:  python predict.py path/to/photo.jpg [more.jpg ...]
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

import config as C


def main(paths):
    interp = tf.lite.Interpreter(model_path=os.path.join(C.OUT_DIR, "model_int8.tflite"))
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = i_d["quantization"]
    s_out, z_out = o_d["quantization"]
    for p in paths:
        img = Image.open(p).convert("L").resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR)
        px = np.asarray(img, dtype=np.float32)
        q = np.clip(np.round(px / s_in + z_in), -128, 127).astype(np.int8)[None, ..., None]
        interp.set_tensor(i_d["index"], q)
        interp.invoke()
        raw = interp.get_tensor(o_d["index"])[0]
        probs = (raw.astype(np.float32) - z_out) * s_out
        k = int(probs.argmax())
        detail = "  ".join(f"{c}={pr:.2f}" for c, pr in zip(C.CLASSES, probs))
        print(f"{p}: {C.CLASSES[k].upper()} ({probs[k]:.0%})   [{detail}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
