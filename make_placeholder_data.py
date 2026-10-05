"""
Generate FAKE briquette images so the whole pipeline can be tested before real photos exist.

These are NOT real charcoal. They only prove the code works end to end
(load -> train -> quantize -> evaluate -> export). Delete data/ and replace with
real photos from the ESP32-S3 camera before training for the thesis.

Usage:  python make_placeholder_data.py --per-class 300
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import CLASSES, DATA_DIR, IMG_SIZE

S = IMG_SIZE * 2  # draw at 2x then downscale for smoother edges


def base_briquette(rng, tone):
    """Dark round briquette (top view, with center hole) on a light tray, with random lighting."""
    bg = int(rng.integers(150, 210))
    img = Image.new("L", (S, S), bg)
    d = ImageDraw.Draw(img)
    cx = S / 2 + rng.uniform(-12, 12)
    cy = S / 2 + rng.uniform(-12, 12)
    r = rng.uniform(0.32, 0.40) * S
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(tone))
    hr = r * rng.uniform(0.15, 0.22)
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=bg)
    return img, (cx, cy, r, hr)


def add_texture(arr, rng, strength):
    noise = rng.normal(0, strength, arr.shape)
    return arr + noise


def make_good(rng):
    img, _ = base_briquette(rng, rng.integers(35, 60))
    return img, 6


def make_cracked(rng):
    img, (cx, cy, r, hr) = base_briquette(rng, rng.integers(35, 60))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(1, 4)):
        # crack runs from near the hole outward with a jagged path
        ang = rng.uniform(0, 2 * np.pi)
        x, y = cx + np.cos(ang) * hr * 1.1, cy + np.sin(ang) * hr * 1.1
        pts = [(x, y)]
        length = rng.uniform(0.5, 1.0) * (r - hr)
        steps = 8
        for _ in range(steps):
            ang += rng.uniform(-0.5, 0.5)
            x += np.cos(ang) * length / steps
            y += np.sin(ang) * length / steps
            pts.append((x, y))
        shade = int(rng.choice([rng.integers(5, 20), rng.integers(110, 160)]))
        d.line(pts, fill=shade, width=int(rng.integers(2, 5)))
    return img, 6


def make_under_carbonized(rng):
    # lighter overall tone + blotchy un-burned patches
    img, (cx, cy, r, hr) = base_briquette(rng, rng.integers(60, 85))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(3, 8)):
        a = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(hr * 1.3, r * 0.8)
        px, py = cx + np.cos(a) * dist, cy + np.sin(a) * dist
        pr = rng.uniform(6, 18)
        d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=int(rng.integers(95, 135)))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    return img, 9


MAKERS = {"good": make_good, "cracked": make_cracked, "under_carbonized": make_under_carbonized}


def finish(img, rng, tex):
    img = img.rotate(rng.uniform(0, 360), fillcolor=int(np.array(img)[0, 0]))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = add_texture(arr, rng, tex)
    # random lighting gradient + brightness/contrast jitter (simulates imperfect lighting)
    gx = np.linspace(-1, 1, IMG_SIZE)[None, :] * rng.uniform(-15, 15)
    gy = np.linspace(-1, 1, IMG_SIZE)[:, None] * rng.uniform(-15, 15)
    arr = (arr + gx + gy) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        os.makedirs(folder, exist_ok=True)
        for i in range(args.per_class):
            img, tex = MAKERS[cls](rng)
            finish(img, rng, tex).save(os.path.join(folder, f"placeholder_{i:04d}.jpg"), quality=90)
        print(f"{cls:>18}: {args.per_class} images -> {folder}")


if __name__ == "__main__":
    main()
