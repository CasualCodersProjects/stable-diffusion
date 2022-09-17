
from ldm.outpainting_utils import *
from ldm.generate import Generate
import numpy as np
import scipy
from scipy.spatial import cKDTree

device = 'mps'


# print("Loading image...")
# # open the image
# orig_img = Image.open("./test_images/airship_partial.png")

# # convert to numpy array
# img = np.array(orig_img)

# # get the alpha channel as a mask
# mask = img[:, :, 3]

# # output the transparent part
# Image.fromarray(mask).save("./test_images/cutout_mask.png")

# # get the RGB channels
# img = img[:, :, 0:3]

# # print("Processing image and mask...")
# img, mask = edge_pad(img, mask)

# img = Image.fromarray(img)
# newMask = Image.new("RGBA", img.size)

# # make the alpha channel of newMask the same as the mask


# print(mask.getpixel((0, 0)))
# print(mask.getpixel((256, 256)))
# print(mask.getpixel((511, 511)))

# print("Saving image and mask...")
# img.save("./test_images/cutout_processed.png")
# mask.save("./test_images/cutout_mask_processed.png")

print("Loading model...")
sd = Generate(device_type=device)

prompt = "a burning airship over a destroyed city, 4k, nightmare, vivid colors, trending on artstation"

print("Generating image...")
# generate the inpainted image
images = sd.img2img(prompt=prompt, steps=50, init_img="./test_images/airship_processed.png", init_mask="./test_images/airship_partial.png")

# # load the generated image to RAM
# image_path = images[0][0]

# # open the image
# img = Image.open(image_path)

