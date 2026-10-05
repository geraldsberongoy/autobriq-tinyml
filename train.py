"""
AUTOBRIQ TinyML training pipeline.

  data/<class>/*.jpg  ->  train tiny CNN  ->  int8 quantize  ->  evaluate on held-out test set
                      ->  export C header for the ESP32-S3 (TensorFlow Lite Micro)

Usage:  python train.py
Outputs (in output/):
  model.keras            full-precision Keras model
  model_float.tflite     float32 TFLite (reference)
  model_int8.tflite      int8 TFLite  <-- this is what runs on the ESP32-S3
  model_data.h / .cc     the int8 model as a C array + labels + input quantization constants
  report.json            accuracy, per-class metrics, confusion matrix, size, memory estimate
  confusion_matrix.png   for the thesis paper
  training_curves.png    for the thesis paper
"""
import glob
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import config as C

keras = tf.keras
layers = keras.layers


# ---------------------------------------------------------------- data
def load_dataset():
    xs, ys, paths = [], [], []
    for idx, cls in enumerate(C.CLASSES):
        files = sorted(
            f for ext in ("jpg", "jpeg", "png", "bmp")
            for f in glob.glob(os.path.join(C.DATA_DIR, cls, f"*.{ext}"))
            + glob.glob(os.path.join(C.DATA_DIR, cls, f"*.{ext.upper()}"))
        )
        if not files:
            raise SystemExit(f"No images found in {C.DATA_DIR}/{cls}/  (expected .jpg/.png)")
        for f in files:
            img = Image.open(f).convert("L").resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR)
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(idx)
            paths.append(f)
        print(f"  {cls:>18}: {len(files)} images")
    x = np.stack(xs)[..., None]  # (N, 96, 96, 1) uint8
    return x, np.array(ys), np.array(paths)


def split(x, y, p):
    x_tr, x_te, y_tr, y_te, p_tr, p_te = train_test_split(
        x, y, p, test_size=C.TEST_SPLIT, stratify=y, random_state=C.SEED)
    val_frac = C.VAL_SPLIT / (1 - C.TEST_SPLIT)
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=val_frac, stratify=y_tr, random_state=C.SEED)
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te, p_te)


augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5, fill_mode="reflect"),
    layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
    layers.RandomBrightness(0.1, value_range=(0, 255)),
], name="augment")


def make_ds(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y))
    if training:
        ds = ds.shuffle(len(x), seed=C.SEED)
        ds = ds.batch(C.BATCH_SIZE).map(
            lambda a, b: (tf.clip_by_value(augment(a, training=True), 0, 255), b),
            num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(C.BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- model
def build_model(n_classes):
    """MobileNet-style depthwise-separable CNN. Every op here is supported by TFLite Micro + ESP-NN."""
    w = lambda c: max(8, int(c * C.WIDTH))

    def sep_block(x, filters, stride):
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU(6.0)(x)
        x = layers.Conv2D(filters, 1, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        return layers.ReLU(6.0)(x)

    inp = keras.Input((C.IMG_SIZE, C.IMG_SIZE, C.CHANNELS), name="image")  # raw pixels 0..255
    x = layers.Rescaling(1.0 / 255)(inp)
    x = layers.Conv2D(w(8), 3, strides=2, padding="same", use_bias=False)(x)  # 48x48
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU(6.0)(x)
    x = sep_block(x, w(16), 1)
    x = sep_block(x, w(32), 2)   # 24x24
    x = sep_block(x, w(32), 1)
    x = sep_block(x, w(64), 2)   # 12x12
    x = sep_block(x, w(64), 1)
    x = sep_block(x, w(128), 2)  # 6x6
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="autobriq_net")


def count_macs(model):
    macs = 0
    for l in model.layers:
        if isinstance(l, layers.DepthwiseConv2D):
            _, h, w_, c = l.output.shape
            macs += h * w_ * c * l.kernel_size[0] * l.kernel_size[1]
        elif isinstance(l, layers.Conv2D):
            _, h, w_, c = l.output.shape
            cin = l.input.shape[-1]
            macs += h * w_ * c * cin * l.kernel_size[0] * l.kernel_size[1]
        elif isinstance(l, layers.Dense):
            macs += l.input.shape[-1] * l.units
    return int(macs)


def recalibrate_batchnorm(model, x, max_images=512):
    """Re-estimate BatchNorm mean/variance from un-augmented training images (single large batch)."""
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(x), size=min(max_images, len(x)), replace=False)
    bns = [l for l in model.layers if isinstance(l, layers.BatchNormalization)]
    old = [l.momentum for l in bns]
    for l in bns:
        l.momentum = 0.0  # stats = exactly this batch's stats
    model(tf.constant(x[idx].astype("float32")), training=True)
    for l, m in zip(bns, old):
        l.momentum = m


# ---------------------------------------------------------------- quantization
def to_int8_tflite(model, x_rep):
    def rep():
        for i in range(min(300, len(x_rep))):
            yield [x_rep[i:i + 1].astype("float32")]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(tfl_bytes, x):
    interp = tf.lite.Interpreter(model_content=tfl_bytes)
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = []
    for img in x:
        inp = img[None].astype("float32")
        if i_d["dtype"] == np.int8:
            s, z = i_d["quantization"]
            inp = np.clip(np.round(inp / s + z), -128, 127).astype(np.int8)
        interp.set_tensor(i_d["index"], inp)
        interp.invoke()
        out = interp.get_tensor(o_d["index"])[0]
        if o_d["dtype"] == np.int8:
            s, z = o_d["quantization"]
            out = (out.astype("float32") - z) * s
        preds.append(out)
    return np.array(preds), i_d, o_d, interp


def estimate_arena_bytes(interp):
    """Rough peak activation memory: largest (inputs+outputs) of any op, non-constant tensors only."""
    tensors = {t["index"]: t for t in interp.get_tensor_details()}
    peak = 0
    for op in interp._get_ops_details():
        total = 0
        for idx in list(op["inputs"]) + list(op["outputs"]):
            if idx < 0 or idx not in tensors:
                continue
            t = tensors[idx]
            if t["name"].startswith(("tfl.pseudo_qconst", "arith.constant")) or "kernel" in t["name"].lower():
                continue
            total += int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
        peak = max(peak, total)
    return peak


# ---------------------------------------------------------------- export
def export_c(tfl_bytes, i_d, path_h, path_cc):
    s, z = i_d["quantization"]
    labels = ", ".join(f'"{c}"' for c in C.CLASSES)
    with open(path_h, "w") as f:
        f.write(f"""// Auto-generated by train.py -- do not edit by hand.
#pragma once
#include <stdint.h>

extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

constexpr int kImgSize = {C.IMG_SIZE};
constexpr int kChannels = {C.CHANNELS};
constexpr int kNumClasses = {len(C.CLASSES)};
static const char* kLabels[kNumClasses] = {{{labels}}};

// Input quantization: int8_value = round(pixel / kInputScale) + kInputZeroPoint
// With scale = 1.0 and zero point = -128 this is simply: int8_value = pixel - 128
constexpr float kInputScale = {s:.8f}f;
constexpr int kInputZeroPoint = {int(z)};
""")
    with open(path_cc, "w") as f:
        f.write('// Auto-generated by train.py -- int8 model for TensorFlow Lite Micro.\n')
        f.write('#include "model_data.h"\n\n')
        f.write("alignas(16) const unsigned char g_model_data[] = {\n")
        for i in range(0, len(tfl_bytes), 16):
            f.write("  " + ", ".join(f"0x{b:02x}" for b in tfl_bytes[i:i + 16]) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int g_model_data_len = {len(tfl_bytes)};\n")


def plot_cm(cm, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (int8 model, test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_history(h, path):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), dpi=150)
    axs[0].plot(h["loss"], label="train"); axs[0].plot(h["val_loss"], label="val"); axs[0].set_title("Loss")
    axs[1].plot(h["accuracy"], label="train"); axs[1].plot(h["val_accuracy"], label="val"); axs[1].set_title("Accuracy")
    for a in axs:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    tf.keras.utils.set_random_seed(C.SEED)
    os.makedirs(C.OUT_DIR, exist_ok=True)

    print("Loading images...")
    x, y, p = load_dataset()
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te, p_te) = split(x, y, p)
    print(f"Split: train={len(x_tr)}  val={len(x_va)}  test={len(x_te)}")

    counts = np.bincount(y_tr, minlength=len(C.CLASSES))
    class_weight = {i: len(y_tr) / (len(C.CLASSES) * max(c, 1)) for i, c in enumerate(counts)}

    model = build_model(len(C.CLASSES))
    steps = int(np.ceil(len(x_tr) / C.BATCH_SIZE)) * C.EPOCHS
    lr = keras.optimizers.schedules.CosineDecay(C.LEARNING_RATE, decay_steps=steps, alpha=0.02)
    model.compile(keras.optimizers.AdamW(lr, weight_decay=C.WEIGHT_DECAY),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary(print_fn=lambda s: None)
    print(f"Params: {model.count_params():,}   MACs/inference: {count_macs(model):,}")

    hist = model.fit(
        make_ds(x_tr, y_tr, True), validation_data=make_ds(x_va, y_va, False),
        epochs=C.EPOCHS, class_weight=class_weight, verbose=2)

    # Augmentation skews BatchNorm's running statistics; recompute them on clean images.
    # On small datasets this step is the difference between ~80% and ~98% validation accuracy.
    val_before = model.evaluate(make_ds(x_va, y_va, False), verbose=0)[1]
    recalibrate_batchnorm(model, x_tr)
    val_after = model.evaluate(make_ds(x_va, y_va, False), verbose=0)[1]
    print(f"Val accuracy before BN recalibration: {val_before:.3f}  after: {val_after:.3f}")
    model.save(os.path.join(C.OUT_DIR, "model.keras"))
    plot_history(hist.history, os.path.join(C.OUT_DIR, "training_curves.png"))

    float_acc = model.evaluate(make_ds(x_te, y_te, False), verbose=0)[1]

    print("Converting to TFLite (float + int8)...")
    float_tfl = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    open(os.path.join(C.OUT_DIR, "model_float.tflite"), "wb").write(float_tfl)
    int8_tfl = to_int8_tflite(model, x_tr)
    open(os.path.join(C.OUT_DIR, "model_int8.tflite"), "wb").write(int8_tfl)

    print("Evaluating int8 model on test set (this is what the ESP32 will do)...")
    probs, i_d, o_d, interp = run_tflite(int8_tfl, x_te)
    pred = probs.argmax(1)
    int8_acc = float((pred == y_te).mean())
    cm = confusion_matrix(y_te, pred, labels=list(range(len(C.CLASSES))))
    rep = classification_report(y_te, pred, target_names=C.CLASSES, output_dict=True, zero_division=0)
    plot_cm(cm, os.path.join(C.OUT_DIR, "confusion_matrix.png"))

    wrong = [{"file": str(f), "actual": C.CLASSES[a], "predicted": C.CLASSES[b]}
             for f, a, b in zip(p_te, y_te, pred) if a != b]

    export_c(int8_tfl, i_d, os.path.join(C.OUT_DIR, "model_data.h"), os.path.join(C.OUT_DIR, "model_data.cc"))
    arena = estimate_arena_bytes(interp)

    report = {
        "classes": C.CLASSES,
        "dataset": {"train": len(x_tr), "val": len(x_va), "test": len(x_te),
                    "per_class_total": np.bincount(y, minlength=len(C.CLASSES)).tolist()},
        "model": {"params": int(model.count_params()), "macs_per_inference": count_macs(model),
                  "int8_tflite_bytes": len(int8_tfl), "float_tflite_bytes": len(float_tfl),
                  "est_peak_activation_bytes": int(arena),
                  "recommended_tensor_arena_bytes": int(max(64 * 1024, 2 * arena))},
        "input_quantization": {"scale": float(i_d["quantization"][0]), "zero_point": int(i_d["quantization"][1])},
        "val_accuracy": {"before_bn_recalibration": round(float(val_before), 4),
                         "after_bn_recalibration": round(float(val_after), 4)},
        "test_accuracy": {"float32": round(float(float_acc), 4), "int8": round(int8_acc, 4)},
        "per_class": {c: {k: round(v, 4) for k, v in rep[c].items()} for c in C.CLASSES},
        "confusion_matrix": cm.tolist(),
        "misclassified": wrong,
    }
    json.dump(report, open(os.path.join(C.OUT_DIR, "report.json"), "w"), indent=2)

    print("\n================ RESULTS ================")
    print(f"Test accuracy  float32: {float_acc:.3f}   int8: {int8_acc:.3f}")
    print(f"int8 model size: {len(int8_tfl)/1024:.1f} KB   est. peak activations: {arena/1024:.1f} KB")
    print(f"Input quantization: scale={i_d['quantization'][0]}  zero_point={i_d['quantization'][1]}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("   " + "  ".join(f"{c[:8]:>8}" for c in C.CLASSES))
    for c, row in zip(C.CLASSES, cm):
        print(f"{c[:8]:>8} " + "  ".join(f"{v:>8}" for v in row))
    print(f"Misclassified: {len(wrong)} (listed in report.json)")
    print(f"All outputs in ./{C.OUT_DIR}/")


if __name__ == "__main__":
    main()
