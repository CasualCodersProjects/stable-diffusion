import sys
import os

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
import numpy as np
from PIL import Image
import scipy
from scipy.spatial import cKDTree

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)

device = 'cuda'
text2img = None
inpaint = None

# from https://github.com/lkwq007/stablediffusion-infinity/blob/d654ca6df95cd982764ec9012c3fb7f9f9dd0a07/utils.py#L72
def edge_pad(img, mask, mode=1):
    if mode == 0:
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res0 = 1 - nmask
        res1 = nmask
        p0 = np.stack(res0.nonzero(), axis=0).transpose()
        p1 = np.stack(res1.nonzero(), axis=0).transpose()
        min_dists, min_dist_idx = cKDTree(p1).query(p0, 1)
        loc = p1[min_dist_idx]
        for (a, b), (c, d) in zip(p0, loc):
            img[a, b] = img[c, d]
    elif mode == 1:
        record = {}
        kernel = [[1] * 3 for _ in range(3)]
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res = scipy.signal.convolve2d(
            nmask, kernel, mode="same", boundary="fill", fillvalue=1
        )
        res[nmask < 1] = 0
        res[res == 9] = 0
        res[res > 0] = 1
        ylst, xlst = res.nonzero()
        queue = [(y, x) for y, x in zip(ylst, xlst)]
        # bfs here
        cnt = res.astype(np.float32)
        acc = img.astype(np.float32)
        step = 1
        h = acc.shape[0]
        w = acc.shape[1]
        offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            target = []
            for y, x in queue:
                val = acc[y][x]
                for yo, xo in offset:
                    yn = y + yo
                    xn = x + xo
                    if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                        if record.get((yn, xn), step) == step:
                            acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                            cnt[yn][xn] += 1
                            acc[yn][xn] /= cnt[yn][xn]
                            if (yn, xn) not in record:
                                record[(yn, xn)] = step
                                target.append((yn, xn))
            step += 1
            queue = target
        img = acc.astype(np.uint8)
    else:
        nmask = mask.copy()
        ylst, xlst = nmask.nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        content = img[yt : yb + 1, xt : xb + 1]
        img = np.pad(
            content,
            ((yt, mask.shape[0] - yb - 1), (xt, mask.shape[1] - xb - 1), (0, 0)),
            mode="edge",
        )
    return img, mask

print("Loading models...")
if sys.platform == 'darwin':
    device = 'mps'
    # Create a pipeline
    text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HUGGINGFACE_TOKEN).to(device)
    inpaint = StableDiffusionInpaintPipeline(
                vae=text2img.vae,
                text_encoder=text2img.text_encoder,
                tokenizer=text2img.tokenizer,
                unet=text2img.unet,
                scheduler=text2img.scheduler,
                safety_checker=text2img.safety_checker,
                feature_extractor=text2img.feature_extractor
    ).to(device)
else:
    # Create a pipeline
    text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=HUGGINGFACE_TOKEN).to(device)
    inpaint = StableDiffusionInpaintPipeline(
                vae=text2img.vae,
                text_encoder=text2img.text_encoder,
                tokenizer=text2img.tokenizer,
                unet=text2img.unet,
                scheduler=text2img.scheduler,
                safety_checker=text2img.safety_checker,
                feature_extractor=text2img.feature_extractor
    ).to(device)

prompt = "warm up thing"
# First-time "warmup" pass
print("Warming up...")
_ = text2img(prompt, num_inference_steps=1)

# open the image
img = Image.open("./test_images/airship_partial.png")

# convert to numpy array
img = np.array(img)

# get the alpha channel as a mask
mask = img[:, :, 3]

# make the lower left corner of the mask transparent
mask[:100, :100] = 1

# output the transparent part
Image.fromarray(mask).save("./test_images/airship_mask.png")

# get the RGB channels
img = img[:, :, 0:3]

img, mask = edge_pad(img, mask)

img = Image.fromarray(img)
mask = Image.fromarray(mask)

images = inpaint(
    prompt="a burning airship over a destroyed city, 4k, nightmare, vivid colors, trending on artstation",
    init_image=img,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.75,
)["sample"]

out = np.zeros((512, 512, 4), dtype=np.uint8)
out[:, :, 0:3] = np.array(
    images[0].resize(
        (512, 512),
        resample=Image.Resampling.LANCZOS,
    )
)
out[:, :, -1] = 255

Image.fromarray(out).save("./test_images/airship_out_1.png")
