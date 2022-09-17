import sys
import argparse
from ldm.outpainting_utils import edge_pad
from ldm.generate import Generate
from PIL import Image
import numpy as np

device = 'mps' if sys.platform == 'darwin' else 'cuda'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Outpainting desktop backgrounds with stable diffusion')
    parser.add_argument('--image', type=str, default=None, help='path to image')
    parser.add_argument('--height', type=int, default=2160, help='height of the output image')
    parser.add_argument('--width', type=int, default=3840, help='width of the output image')
    parser.add_argument('--steps', type=int, default=50, help='number of diffusion steps')
    # parser.add_argument('--upscale_factor', type=int, default=2, help='upscale factor. needs to be 1, 2, or 4')
    parser.add_argument('--prompt', type=str, help='prompt for the image')
    args = parser.parse_args()

    height = args.height
    width = args.width
    
    image = args.image

    if args.prompt is None:
        print("Please provide a prompt")
        sys.exit(1)

    sd = Generate(device_type=device)

    # if no image specified, use the prompt to generate one
    if image is None:
        output = sd.txt2img(args.prompt, steps=args.steps, height=512, width=512)
        image = output[0][0]
    
    # do more stuff here

    

# print("Loading image...")
# # open the image
# orig_img = Image.open("./test_images/airship_corner.png")

# # convert to numpy array
# img = np.array(orig_img)

# # get the alpha channel as a mask
# mask = img[:, :, 3]

# # output the transparent part
# Image.fromarray(mask).save("./test_images/airship_corner_mask.png")

# # get the RGB channels
# img = img[:, :, 0:3]

# # print("Processing image and mask...")
# img, mask = edge_pad(img, mask)

# # save the image and make to disk
# Image.fromarray(img).save("./test_images/airship_corner_processed.png")
# Image.fromarray(mask).save("./test_images/airship_corner_mask_processed.png")


# img = Image.fromarray(img)
# newMask = Image.new("RGBA", img.size)

# # make the alpha channel of newMask the same as the mask


# print(mask.getpixel((0, 0)))
# print(mask.getpixel((256, 256)))
# print(mask.getpixel((511, 511)))

# print("Saving image and mask...")
# img.save("./test_images/cutout_processed.png")
# mask.save("./test_images/cutout_mask_processed.png")

# print("Loading model...")
# sd = Generate(device_type=device)

# prompt = "a burning airship over a destroyed city, 4k, nightmare, vivid colors, trending on artstation"

# print("Generating image...")
# # generate the inpainted image
# images = sd.img2img(prompt=prompt, steps=50, init_img="./test_images/airship_corner_processed.png", init_mask="./test_images/airship_corner.png")

# # load the generated image to RAM
# image_path = images[0][0]

# # open the image
# img = Image.open(image_path)

