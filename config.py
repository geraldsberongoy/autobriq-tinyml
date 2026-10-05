"""Shared settings for the AUTOBRIQ TinyML pipeline. Change these in ONE place."""

# Class names = folder names inside data/. Order matters: index 0, 1, 2 is what the ESP32 outputs.
CLASSES = ["good", "cracked", "under_carbonized"]

IMG_SIZE = 96          # 96x96 pixels (matches what the ESP32-S3 will feed the model)
CHANNELS = 1           # 1 = grayscale

DATA_DIR = "data"      # data/good/*.jpg, data/cracked/*.jpg, data/under_carbonized/*.jpg
OUT_DIR = "output"     # trained model, int8 .tflite, C header, reports go here

SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4

# Width multiplier for the network. 2.0 = ~73k params (~100 KB int8). Lower = smaller/faster, higher = more accurate.
WIDTH = 2.0
