from ldm.generate import Generate
from PIL import Image, ImageStat
import os
import time
import math

output_dir = os.path.join(os.getcwd(), 'outputs', 'backgrounds')
working_dir = os.path.join(output_dir, 'tmp')


def main(prompt, gen_width=1024, gen_height=512, vertical=False, scale_factor=4, scale_strength=0.75, height=2160, width=3840, iterations=1, output_name=''):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    sd = Generate()

    # generate an image using the prompt
    output = sd.txt2img(prompt=prompt, height=gen_height,
                        width=gen_width, upscale=(scale_factor, scale_strength))

    img_path, _ = output[0]

    # open the image using PIL
    img = Image.open(img_path)

    output_width, output_height = img.size

    # crop 16:9 samples
    frac = 16/9

    crops = {}

    if vertical:
        crop_height = math.floor(output_width * frac)
        crops = {
            'top': img.crop((0, 0, output_width, crop_height)),
            'bottom': img.crop((0, output_height - crop_height, output_width, crop_height)),
            'center': img.crop((0, math.floor((output_height - crop_height) / 2), output_width, crop_height))
        }
    else:
        crop_width = math.floor(output_height * frac)
        crops = {
            'left': img.crop((0, 0, crop_width, output_height)),
            'right': img.crop((output_width - crop_width, 0, crop_width, output_height)),
            'center': img.crop((math.floor((output_width - crop_width) / 2), 0, crop_width, output_height))
        }

    stddev_dict = {}
    for crop in crops:
        stddev_dict[crop] = ImageStat.Stat(crops[crop]).stddev

    # get the one with the highest stddev
    best_crop = max(stddev_dict, key=stddev_dict.get)

    size = (2160, 3840) if vertical and height == 2160 and width == 3840 else (
        width, height)

    # resized to 2160p
    img = crops[best_crop].resize(size, 1)

    # save the image
    img.convert('RGB').save(os.path.join(
        output_dir, f'background.{time.time()}.jpg'))


if __name__ == '__main__':
    prompt = 'seductive anime girl as a garden fairy l, hourglass slim figure, red hair hair, attractive features, tight fitted tank top, body portrait, slight smile, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by WLOP and greg rutkowski and alphonse mucha and artgerm'
    height = 2160
    width = 3840
    vertical = False
    main(prompt, width=width, height=height, vertical=vertical)
