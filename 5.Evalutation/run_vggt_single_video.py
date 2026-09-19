import glob
import os
import subprocess

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

VIDEO = "/scratch/ciro/tnv2_demo/match1.mp4"
FRAME_DIR = "/scratch/ciro/tnv2_demo/vggt_frames"
OUT_PATH = "/scratch/ciro/tnv2_demo/match1_vggt.npz"
FRAME_STRIDE = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print("device:", device, "dtype:", dtype)

os.makedirs(FRAME_DIR, exist_ok=True)
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()
print("model loaded")

subprocess.run(
    ["ffmpeg", "-y", "-i", VIDEO, "-vf", r"select=not(mod(n\,10))", "-vsync", "vfr",
     os.path.join(FRAME_DIR, "frame_%05d.png"), "-hide_banner", "-loglevel", "error"],
    check=True,
)
image_names = sorted(glob.glob(os.path.join(FRAME_DIR, "frame_*.png")))
print(f"{len(image_names)} frames extracted")

images = load_and_preprocess_images(image_names).to(device)
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
        images_b = images[None]
        aggregated_tokens_list, ps_idx = model.aggregator(images_b)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_b.shape[-2:])

extrinsic_np = extrinsic[0].float().cpu().numpy()
intrinsic_np = intrinsic[0].float().cpu().numpy()
sampled_frame_indices = np.arange(len(image_names)) * FRAME_STRIDE

np.savez(OUT_PATH, extrinsic=extrinsic_np, intrinsic=intrinsic_np,
         sampled_frame_indices=sampled_frame_indices, num_sampled_frames=len(image_names))
print("saved", OUT_PATH)
print("intrinsic shape:", intrinsic_np.shape)
# closest sampled frame to our GT frame_index=125
target = 125
closest_i = int(np.argmin(np.abs(sampled_frame_indices - target)))
print(f"closest sampled frame to {target}: idx={sampled_frame_indices[closest_i]}, K=\n{intrinsic_np[closest_i]}")
