# AUTOBRIQ TinyML: Briquette Quality Classifier

Offline image classifier for the ESP32-S3. It takes a 96×96 grayscale photo of a briquette and outputs one of three classes:

| Index | Label | Meaning |
|---|---|---|
| 0 | `good` | Uniform, fully carbonized, no cracks |
| 1 | `cracked` | Visible cracks or fractures |
| 2 | `under_carbonized` | Lighter or uneven patches (not fully burned) |

Training runs on a laptop. The resulting int8 model runs on the ESP32-S3 with **no internet**.

## Setup (once)

```bash
pip install -r requirements.txt
```

## Test the pipeline with fake images

```bash
python make_placeholder_data.py --per-class 300   # creates data/ with synthetic briquettes
python train.py                                    # ~2-3 min on a laptop CPU
python predict.py data/cracked/placeholder_0001.jpg
```

The placeholder images are **not real charcoal**. They only prove the code works.

## Train on real photos

1. Delete the `data/` folder.
2. Put real photos from the ESP32-S3 camera into:
   ```
   data/good/
   data/cracked/
   data/under_carbonized/
   ```
   Any filenames work, as .jpg or .png. Aim for **150–300+ per class**, shot under the **same lighting, distance and camera settings** the machine will use.
3. Run `python train.py`.
4. Check `output/report.json` and `output/confusion_matrix.png`.

To change or add classes, edit `CLASSES` in `config.py` and make matching folders.

## Outputs (`output/`)

| File | Use |
|---|---|
| `model_int8.tflite` | The model that runs on the ESP32-S3 |
| `model_data.h` / `model_data.cc` | The same model as a C array. Copy both into the Arduino/ESP-IDF project |
| `report.json` | Accuracy, per-class precision/recall/F1, confusion matrix, misclassified files, memory sizes |
| `confusion_matrix.png`, `training_curves.png` | Figures for the thesis paper |
| `model.keras`, `model_float.tflite` | Full-precision reference models |

## Firmware notes (for the ESP32-S3 side)

- **Input:** 96×96 grayscale, 1 byte per pixel. Convert each pixel with `int8 = pixel - 128` (scale 1.0, zero point -128 are in `model_data.h`).
- **Ops needed** (register these in `MicroMutableOpResolver`): `CONV_2D`, `DEPTHWISE_CONV_2D`, `MEAN`, `FULLY_CONNECTED`, `SOFTMAX`.
- **Tensor arena:** about 220 KB recommended (put it in PSRAM, or in internal RAM for speed).
- **Model size:** about 112 KB flash.
- **Compute:** about 10.9 M MACs per inference. Use the `esp-tflite-micro` library, which includes ESP-NN optimizations for the S3.
- **Output:** 3 int8 scores. The highest one is the predicted class. Dequantize with the output scale and zero point if you need the confidence in %.

## Tested results (placeholder data, 900 images, 15% held-out test set)

| Metric | Value |
|---|---|
| Test accuracy, float32 | 97.0% |
| Test accuracy, int8 (on-device equivalent) | 96.3–97.0% |
| Parameters | 73,283 |
| int8 model size | 112 KB |

Re-run on a second, independently generated dataset: 97.0% int8.
Real-photo accuracy will differ. Report the numbers from your own `report.json`.

## Tuning knobs (`config.py`)

- `WIDTH`: network width. 2.0 is the default (73k params). Use 1.0 for a smaller, faster model; try 3.0 if accuracy is low on real data.
- `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `BATCH_SIZE`: training settings.
- Augmentation (flip, rotation, small shift, brightness) is in `train.py` (`augment`).

## Method summary (for the methodology chapter)

- **Architecture:** MobileNet-style depthwise-separable CNN (1 standard conv + 6 depthwise-separable blocks, global average pooling, dropout 0.3, softmax).
- **Training:** AdamW (lr 2e-3, weight decay 1e-4), cosine-decay schedule, 50 epochs, class-balanced weights, stratified 70/15/15 split.
- **BatchNorm recalibration:** after training, BatchNorm statistics are re-estimated on un-augmented training images. Augmentation otherwise skews them; this raised validation accuracy from about 76–91% to about 98.5% in testing.
- **Quantization:** full-integer post-training quantization (int8 weights, activations, input and output) using 300 representative training images. Accuracy is evaluated on the int8 model itself.
