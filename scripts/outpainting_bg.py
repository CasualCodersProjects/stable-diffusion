import random
import time
import os
from typing import Tuple
import shutil
from ldm.outpainting_utils import edge_pad
from ldm.generate import Generate
from PIL import Image
from PIL.Image import Image as ImageClass
import numpy as np

# because of the way esrgan's upscaler is implemented, we can't make the
# script into a CLI tool easily. For now we'll just edit the script.
output_dir = os.path.join(os.getcwd(), 'outputs', 'outpainting')
working_dir = os.path.join(output_dir, 'tmp')
height = 512
width = 512


def edgify(img: np.ndarray) -> ImageClass:
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


def process_image(image_path: str, vertical: bool = False) -> Tuple[str, str, str, str]:
    image = Image.open(image_path).convert('RGBA')

    # if the input image is not 512x512, resize it
    if image.size != (512, 512):
        print('Resizing image to 512x512')
        image = image.resize((512, 512), 1)

    right_side = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    left_side = Image.new('RGBA', (512, 512), (0, 0, 0, 0))

    image_array = np.array(image)
    right_side_array = np.array(right_side)
    left_side_array = np.array(left_side)

    if vertical:
        right_side_array[0:311, :] = image_array[200:511, :]
        left_side_array[200:511, :] = image_array[0:311, :]
    else:
        right_side_array[:, 0:311] = image_array[:, 200:511]
        left_side_array[:, 200:511] = image_array[:, 0:311]

    # edgeify the images
    right_side_edged = edgify(right_side_array)
    left_side_edged = edgify(left_side_array)

    right_side_mask_path = os.path.join(working_dir, 'right_side_mask.png')
    left_side_mask_path = os.path.join(working_dir, 'left_side_mask.png')
    right_side_edged_path = os.path.join(working_dir, 'right_side_edged.png')
    left_side_edged_path = os.path.join(working_dir, 'left_side_edged.png')

    Image.fromarray(right_side_array).save(right_side_mask_path)
    Image.fromarray(left_side_array).save(left_side_mask_path)

    right_side_edged.save(right_side_edged_path)
    left_side_edged.save(left_side_edged_path)

    return right_side_mask_path, left_side_mask_path, right_side_edged_path, left_side_edged_path


def combine_images(right_path: str, left_path: str, vertical=False, jpeg=False) -> Tuple[ImageClass, str, str]:

    right_side_outpainted = Image.open(
        right_path).convert('RGBA')
    left_side_outpainted = Image.open(
        left_path).convert('RGBA')

    image_mode = 'RGB' if jpeg else 'RGBA'

    final_image = None
    if vertical:
        final_image = Image.new(image_mode, (512, 912), (0, 0, 0, 0))
        final_image.paste(right_side_outpainted, (0, 400))
        final_image.paste(left_side_outpainted, (0, 0))
    else:
        final_image = Image.new(image_mode, (912, 512), (0, 0, 0, 0))
        # the left image becomes the left 512 pixels
        final_image.paste(left_side_outpainted, (0, 0))
        # paste the right into the image offset by 400 pixels
        final_image.paste(right_side_outpainted, (400, 0))

    final_ending = 'jpg' if jpeg else 'png'

    final_image_path = os.path.join(
        output_dir, f'outpaint_bg.{time.time()}.{final_ending}')

    final_image_upscaled_path = os.path.join(
        output_dir, f'outpaint_bg.upscaled.{time.time()}.{final_ending}')

    final_image.save(final_image_path)

    return final_image, final_image_path, final_image_upscaled_path


def run_upscale(sd: Generate, input_image: ImageClass, input_seed: int, input_path: str, output_path: str, keep_original=False) -> None:
    # callback for upscaling
    def cb(image, seed, upscaled=None):
        if not upscaled:
            print("Upscaling failed. Saving original image...")
            return
        # convert the image to jpeg
        image.save(output_path)

        if not keep_original:
            os.remove(input_path)

    sd.upscale_and_reconstruct(
        image_list=[(input_image, input_seed)], image_callback=cb, upscale=(4, 0.75))


def outpainting_bg(prompt, img='', steps=50, upscale=True, keep_original=False, clean=True, vertical=False, jpeg=True, sd=None):

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    if not sd:
        sd = Generate()
    # to make a 16:9 image at 512 generation, we only need to expand 200 pixels in either direction

    image_path = img
    seed = random.randint(0, 4_000_000)
    if not img:
        initial_image = sd.txt2img(
            prompt=prompt, height=height, width=width, steps=steps)
        image_path, seed = initial_image[0]
    # open the image using PIL
    print('Processing image...')
    right_side_mask_path, left_side_mask_path, right_side_edged_path, left_side_edged_path = process_image(
        image_path, vertical=vertical)

    print('Outpainting...')

    right_side_outpainted_path = os.path.join(
        working_dir, 'right_side_outpainted.png')
    left_side_outpainted_path = os.path.join(
        working_dir, 'left_side_outpainted.png')

    # now, we need to generate the images
    right_side_outpainted_path = outpaint(sd, right_side_edged_path,
                                          right_side_mask_path, prompt, name=right_side_outpainted_path)
    left_side_outpainted_path = outpaint(sd, left_side_edged_path,
                                         left_side_mask_path, prompt, name=left_side_outpainted_path)

    print("Combining images...")
    final_image, final_image_path, final_image_upscaled_path = combine_images(
        right_side_outpainted_path, left_side_outpainted_path, vertical=vertical, jpeg=jpeg)

    if upscale:
        print("Upscaling...")
        run_upscale(sd, final_image, seed, final_image_path,
                    final_image_upscaled_path, keep_original=keep_original)

    if clean:
        print("Cleaning up....")
        shutil.rmtree(working_dir)

    if not upscale:
        print(f"Done! Image can be found at {final_image_path}")
    else:
        print(
            f"Done! Image can be found at {final_image_upscaled_path}")
        if keep_original:
            print(f"Original image can be found at {final_image_path}")


if __name__ == '__main__':
    prompt = 'a giant dog destroying a city. nightmare. 8k. octane. digital art. artstation. vivid colors.'
    print('Using prompt "', prompt, '"')
    outpainting_bg(prompt, vertical=False, upscale=True, jpeg=True)
