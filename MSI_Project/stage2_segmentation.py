import os
import cv2
import time
import io as ioio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from cellpose import models, plot

from config import MASTER_DIR

def stage2_cellpose_worker(subdir: Path, model):
    stage1_npz, seg_out = subdir / "PREMSI_stage1_ready.npz", subdir / "PREMSI_seg.npy"
    if not stage1_npz.exists() or seg_out.exists(): return

    os.chdir(subdir)
    data = np.load(stage1_npz, allow_pickle=True)
    imgs_cp, image_for_cellpose = data["imgs_cp"], data["image_for_cellpose"]

    if imgs_cp.ndim == 2: imgs_cp = imgs_cp[:, :, np.newaxis]
    h, w, c = imgs_cp.shape

    imgs_3ch = np.concatenate([imgs_cp, np.zeros((h, w, 1), dtype=imgs_cp.dtype)], axis=-1) if c == 2 else (imgs_cp[:, :, :3].copy() if c >= 3 else np.repeat(imgs_cp[:, :, np.newaxis], 3, axis=2))
    img_bgr = cv2.cvtColor(imgs_3ch, cv2.COLOR_RGB2BGR)
    img_bgr = (img_bgr.astype(float) / img_bgr.max() * 255).astype(np.uint8) if img_bgr.max() > 0 else img_bgr.astype(np.uint8)

    masks, _, _ = model.eval(img_bgr, batch_size=64)
    np.save(seg_out, {"masks": masks}, allow_pickle=True)

    def get_seg_overlay(img, msk):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(plot.mask_overlay(img, msk)); ax.axis("off")
        buf = ioio.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig); buf.seek(0)
        return Image.open(buf).convert("RGB")

    try:
        seg1, seg2 = get_seg_overlay(img_bgr, masks), get_seg_overlay(image_for_cellpose, masks)
        combined = Image.new("RGB", (max(seg1.width, seg2.width), seg1.height + seg2.height), (255, 255, 255))
        combined.paste(seg1, (0, 0)); combined.paste(seg2, (0, seg1.height))
        combined.save(subdir / "PREMSI_stage2_audit_segmentation.png")
    except Exception: pass

def run_stage2_cellpose_serial():
    t0 = time.time()
    subdirs = [d for d in MASTER_DIR.iterdir() if d.is_dir()]
    import torch; torch.backends.cudnn.benchmark = True
    model = models.CellposeModel(gpu=True)
    for subdir in subdirs: stage2_cellpose_worker(subdir, model)
    print(f"[S2] DONE in {time.time() - t0:.1f} s")
