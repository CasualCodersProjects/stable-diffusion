import sys
import time
import os
import argparse
from ldm.outpainting_utils import edge_pad
from ldm.generate import Generate
from ldm.dream.devices import choose_torch_device
from PIL import Image
import numpy as np

device = 'mps' if sys.platform == 'darwin' else choose_torch_device()
prompt = 'living alien spaceship over a cyberpunk city, photorealistic, ultra detailed, realistic, 35 mm, photography, high definition, 8k, artstation'
height = 512
width = 512
steps = 50


def edgify(img: np.ndarray) -> Image:
    '''returns an image filled from edge provided'''
    mask = img[:, :, 3]
    rgb_img = img[:, :, 0:3]

    edged_img, _ = edge_pad(rgb_img, mask)

    return Image.fromarray(edged_img)


def outpaint(generator: Generate, image_path: str, mask_path: str, prompt: str, name: str = '', **kwargs) -> str:
    '''outpaints image with mask. outputs image path string. Should use the edged image as the image path and the pixel edge as the mask'''
    output = generator.img2img(
        prompt=prompt, init_img=image_path, init_mask=mask_path, **kwargs)
    if name:
        os.replace(output[0][0], name)
        return name
    return output[0][0]


if __name__ == '__main__':
    sd = Generate(device_type=device)
    # to make a 16:9 image at 512 generation, we only need to expand 200 pixels in either direction
    # first, generate a 512x512 image

    initial_image = sd.txt2img(
        prompt=prompt, height=height, width=width, steps=steps)

    image_path, seed = initial_image[0]
    # open the image using PIL
    print('Loading image...')
    image = Image.open(image_path).convert('RGBA')

    print('Processing image...')
    right_side = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    left_side = Image.new('RGBA', (512, 512), (0, 0, 0, 0))

    image_array = np.array(image)
    right_side_array = np.array(right_side)
    left_side_array = np.array(left_side)

    right_side_array[:, 0:311] = image_array[:, 200:511]
    left_side_array[:, 200:511] = image_array[:, 0:311]

    # edgeify the images
    right_side_edged = edgify(right_side_array)
    left_side_edged = edgify(left_side_array)

    Image.fromarray(right_side_array).save('right_side_mask.png')
    Image.fromarray(left_side_array).save('left_side_mask.png')

    right_side_edged.save('right_side_edged.png')
    left_side_edged.save('left_side_edged.png')

    print('Outpainting...')
    # now, we need to generate the images
    right_side_outpainted = outpaint(sd, 'right_side_edged.png',
                                     'right_side_mask.png', prompt, name='right_side_outpainted.png')
    left_side_outpainted = outpaint(sd, 'left_side_edged.png',
                                    'left_side_mask.png', prompt, name='left_side_outpainted.png')

    right_side_outpainted = 'right_side_outpainted.png'
    left_side_outpainted = 'left_side_outpainted.png'

    # now, we need to combine the images
    right_side_outpainted = Image.open(right_side_outpainted).convert('RGBA')
    left_side_outpainted = Image.open(left_side_outpainted).convert('RGBA')
    final_image = Image.new('RGBA', (912, 512), (0, 0, 0, 0))

    # the left image becomes the left 512 pixels
    final_image.paste(left_side_outpainted, (0, 0))
    # paste the right into the image offset by 200 pixels
    final_image.paste(right_side_outpainted, (400, 0))

    final_image.save('final_image.png')

    # RealESRGAN doesn't work right, need to see if I can get it to work
    # def cb(image, seed, upscaled=None):
    #     image.save('final_image_upscaled.png')

    # sd.upscale_and_reconstruct(
    #     image_list=[(final_image, seed)], image_callback=cb, upscale=(2, 0.75))
