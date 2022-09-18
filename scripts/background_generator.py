from ldm.generate import Generate
from PIL import Image, ImageStat
import os
import time
import math

output_dir = os.path.join(os.getcwd(), 'outputs', 'backgrounds')
working_dir = os.path.join(output_dir, 'tmp')
height = 512
width = 1024


def main(prompt):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    sd = Generate()

    # generate an image using the prompt
    output = sd.txt2img(prompt=prompt, height=height,
                        width=width, upscale=(4, 0.75))

    img_path, _ = output[0]

    # open the image using PIL
    img = Image.open(img_path)

    output_width, output_height = img.size

    # crop 16:9 samples
    frac = 16/9
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

    # resized to 2160p
    img = crops[best_crop].resize((3840, 2160), 1)

    # save the image
    img.convert('RGB').save(os.path.join(
        output_dir, f'background.{time.time()}.jpg'))


if __name__ == '__main__':
    main('seductive anime girl as a garden fairy l, hourglass slim figure, red hair hair, attractive features, tight fitted tank top, body portrait, slight smile, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by WLOP and greg rutkowski and alphonse mucha and artgerm')
