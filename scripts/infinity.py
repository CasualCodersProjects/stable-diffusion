import sys
import time
import os
import argparse
from typing import List
from ldm.outpainting_utils import edge_pad
from ldm.generate import Generate
from ldm.dream.devices import choose_torch_device
from PIL import Image
import numpy as np
import math

device = 'mps' if sys.platform == 'darwin' else choose_torch_device()


class Location:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class PartialImage:
    '''A datatype to keep track of images and where they are located on the canvas
    0,0 is the center of the canvas
    1,0 is one image to the right of the canvas
    -1,0 is one image to the left of the canvas
    0,1 is one image above the canvas
    0,-1 is one image below the canvas

    all images are offset by one pixel on x, one pixel on y, or both to blend with the canvas
    '''

    def __init__(self, image_path: str, location: Location):
        self.image_path = image_path
        self.location = location

    def __str__(self):
        return f'PartialImage(image_path={self.image_path}, location={self.location})'

    def __eq__(self, other):
        return self.image_path == other.image_path and self.location == other.location


def edgify(image_with_edge_pixels: Image) -> Image:
    '''returns an image filled from edge provided'''
    # convert image to numpy array
    img = np.array(image_with_edge_pixels)
    mask = img[:, :, 3]
    rgb_img = img[:, :, 0:3]

    edged_img, _ = edge_pad(rgb_img, mask)

    return Image.fromarray(edged_img)


def outpaint(image_path: str, generator: Generate, mask_path: str, prompt: str, name: str = '', **kwargs) -> str:
    '''outpaints image with mask. outputs image path string. Should use the edged image as the image path and the pixel edge as the mask'''
    output = generator.img2img(
        prompt=prompt, init_img=image_path, init_mask=mask_path, **kwargs)
    if name:
        os.replace(output[0][0], name)
        return name
    return output[0][0]


def create_pixel_edge_image(direction: str, original_above_or_below_img_path: str, original_to_left_or_to_right_img_path: str) -> Image:
    '''returns new image with an edge of pixels from the right side of the original image
    original_img_path is the path to a 512x512 image
    direction is one of the following: above, below, left, right, above_to_left, above_to_right, below_to_left, below_to_right
    '''

    # error checking

    if direction is None:
        raise ValueError('direction cannot be None')

    if direction not in ['above', 'below', 'left', 'right', 'above_to_left', 'above_to_right', 'below_to_left', 'below_to_right']:
        raise ValueError(
            'direction must be one of the following: above, below, left, right, above_to_left, above_to_right, below_to_left, below_to_right')

    if original_above_or_below_img_path is None and original_to_left_or_to_right_img_path is None:
        raise ValueError('at least 1 original_img_path must be provided')

    if direction == 'above' or direction == 'below':
        if original_above_or_below_img_path is None:
            raise ValueError(
                'original_above_or_below_img_path must be provided because of specified direction')

    if direction == 'left' or direction == 'right':
        if original_to_left_or_to_right_img_path is None:
            raise ValueError(
                'original_to_left_or_to_right_img_path must be provided because of specified direction')

    if direction == 'above_to_left' or direction == 'above_to_right' or direction == 'below_to_left' or direction == 'below_to_right':
        if original_above_or_below_img_path is None or original_to_left_or_to_right_img_path is None:
            raise ValueError(
                'original_above_or_below_img_path and original_to_left_or_to_right_img_path must be provided because of specified direction')

    # open supplied images
    if original_above_or_below_img_path is not None:
        original_above_or_below_img = Image.open(
            original_above_or_below_img_path).convert('RGBA')
        original_above_or_below_img_matrix = np.array(
            original_above_or_below_img)

    if original_to_left_or_to_right_img_path is not None:
        original_to_left_or_to_right_img = Image.open(
            original_to_left_or_to_right_img_path).convert('RGBA')
        original_to_left_or_to_right_img_matrix = np.array(
            original_to_left_or_to_right_img)

    # create new image to be returned
    # make sure to change alpha back to zero in prod
    transparent_img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    transparent_img_matrix = np.array(transparent_img)

    if direction == 'above' or direction == 'above_to_left' or direction == 'above_to_right':
        # use when making image above original image
        # top most X axis of original axis becomes bottom x axis of transparent image
        transparent_img_matrix[342:512] = original_above_or_below_img_matrix[0:170]

    elif direction == 'below' or direction == 'below_to_left' or direction == 'below_to_right':
        # use when making image under original image
        # bottom most X axis of original axis becomes top x axis of transparent image
        transparent_img_matrix[0:170] = original_above_or_below_img_matrix[342:512]

    elif direction == 'left' or direction == 'above_to_left' or direction == 'below_to_left':
        # use when making image to the left of original image
        # left most Y axis of original axis becomes right y axis of transparent image
        transparent_img_matrix[:,
                               342:512] = original_to_left_or_to_right_img_matrix[:, 0:170]

    elif direction == 'right' or direction == 'above_to_right' or direction == 'below_to_right':
        # use when making image to the right of original image
        # right most Y axis of original axis becomes left y axis of transparent image
        transparent_img_matrix[:,
                               0:170] = original_to_left_or_to_right_img_matrix[:, 342:512]

    transparent_img_with_pixel_row = Image.fromarray(transparent_img_matrix)

    return transparent_img_with_pixel_row


def create_partial_image_from_edge_filled_img_path(edge_filled_img: Image, location: Location, prompt: str, steps: int, **kwargs) -> str:
    '''most create functions share this logic to create the actual image'''
    print('generating partial image at location', location)
    edge_filled_img.save(
        f'{location.x}x_{location.y}y_pixel_edge.png')
    edge_filled_image = edgify(edge_filled_img)
    edge_filled_image.save(
        f'{location.x}x_{location.y}y_pixel_edge_filled.png')
    created_image_path = outpaint(f'{location.x}x_{location.y}y_pixel_edge_filled.png', sd,
                                  f'{location.x}x_{location.y}y_pixel_edge.png', prompt, steps=steps, height=512, width=512, **kwargs)
    print('done generating partial image at location', location)
    return created_image_path


def create_image_above(below_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='above', original_above_or_below_img_path=below_img_path, original_to_left_or_to_right_img_path=None)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)


def create_image_below(above_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='below', original_above_or_below_img_path=above_img_path, original_to_left_or_to_right_img_path=None)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)


def create_image_to_left(right_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='left', original_above_or_below_img_path=None, original_to_left_or_to_right_img_path=right_img_path)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)


def create_image_to_right(left_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='right', original_above_or_below_img_path=None, original_to_left_or_to_right_img_path=left_img_path)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)


def create_image_above_to_left(below_img_path: str, right_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='above_to_left', original_above_or_below_img_path=below_img_path, original_to_left_or_to_right_img_path=right_img_path)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)


def create_image_above_to_right(below_img_path: str, left_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='above_to_right', original_above_or_below_img_path=below_img_path, original_to_left_or_to_right_img_path=left_img_path)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)


def create_image_below_to_left(above_img_path: str, right_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='below_to_left', original_above_or_below_img_path=above_img_path, original_to_left_or_to_right_img_path=right_img_path)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)


def create_image_below_to_right(above_img_path: str, left_img_path: str, location: Location, prompt: str, **kwargs) -> str:
    pixel_edge_img = create_pixel_edge_image(
        direction='below_to_right', original_above_or_below_img_path=above_img_path, original_to_left_or_to_right_img_path=left_img_path)
    return create_partial_image_from_edge_filled_img_path(edge_filled_img=pixel_edge_img, location=location, prompt=prompt, **kwargs)

# aggregate all the partial images into one image


def aggregate_images(completed_partial_images: List[PartialImage], width, height, upscale_factor=1) -> str:
    print('aggregated images:')
    for image in completed_partial_images:
        print(image)

    final_image = Image.new('RGB', (width, height))
    for partial_image in completed_partial_images:
        img = Image.open(partial_image.image_path)
        x_location = (width/2) - (512/2) + partial_image.location.x*342
        y_location = (height/2) - (512/2) + partial_image.location.y*342*-1
        print(
            f'placing image {partial_image.location} at location', (x_location, y_location))
        final_image.paste(img, (int(x_location), int(y_location)))
        os.remove(partial_image.image_path)
    final_image.show()
    # implement this using RealESRGANer https://github.com/xinntao/Real-ESRGAN/blob/e5763af5749430c9f7389f185cc53f90c4852ed5/realesrgan/utils.py
    # if upscale_factor != 1:
    #     final_image = real_esrgan_upscale(final_image, 0.75, upscale_factor, random.randint(0, 1000000))
    final_image.save(f'./outputs/outpainting/final_image_{time.time()}.png')
    return 'final_image.png'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Outpainting desktop backgrounds with stable diffusion')
    parser.add_argument('--image', type=str, default=None,
                        help='path to image')
    parser.add_argument('--height', type=int, default=1080,
                        help='height of the output image')
    parser.add_argument('--width', type=int, default=1920,
                        help='width of the output image')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of diffusion steps')
    # parser.add_argument('--upscale_factor', type=int, default=1, help='upscale factor. needs to be 1, 2, or 4')
    # parser.add_argument('--gfpgan_scale', type=float, default=0, help='gfpgan scale. float less than 1')
    parser.add_argument('--prompt', type=str, help='prompt for the image')
    args = parser.parse_args()

    image = args.image

    if args.prompt is None:
        print("Please provide a prompt")
        sys.exit(1)

    if not os.path.isdir('./outputs/outpainting'):
        os.mkdir('./outputs/outpainting')

    # if args.upscale_factor not in [1, 2, 4]:
    #     print("Upscale factor needs to be 1, 2, or 4")
    #     sys.exit(1)

    # height = args.height // args.upscale_factor
    # width = args.width // args.upscale_factor

    height = args.height
    width = args.width

    sd = Generate(device_type=device)

    # if no image specified, use the prompt to generate one
    if image is None:
        output = sd.txt2img(args.prompt, steps=args.steps,
                            height=512, width=512)
        image = output[0][0]

    partial_image_0_0 = PartialImage(image, Location(0, 0))

    # how many down, and how many up to go, by default 1 up and 1 down
    max_up_down = math.ceil(((height / 2) - (512/2)) / 342)

    # how many left, and how many right to go, by default 3 left and 3 right
    max_left_right = math.ceil(((width / 2) - (512/2)) / 342)

    print(f'max_up_down: {max_up_down}')
    print(f'max_left_right: {max_left_right}')

    completed_partial_images = []

    # create each partial image that needs to be generated and in what order
    collected_partial_images = []
    collected_partial_images.append(partial_image_0_0)

    for vertical_partial_image in range(max_up_down+1):
        for horizontal_partial_image in range(max_left_right+1):
            if vertical_partial_image == 0 and horizontal_partial_image == 0:
                continue
            if vertical_partial_image == 0:
                collected_partial_images.append(PartialImage(
                    image, Location(horizontal_partial_image, vertical_partial_image)))
                collected_partial_images.append(PartialImage(
                    image, Location(-1*horizontal_partial_image, vertical_partial_image)))
            elif horizontal_partial_image == 0:
                collected_partial_images.append(PartialImage(
                    image, Location(horizontal_partial_image, vertical_partial_image)))
                collected_partial_images.append(PartialImage(image, Location(
                    horizontal_partial_image, -1*vertical_partial_image)))
            else:
                collected_partial_images.append(PartialImage(
                    image, Location(horizontal_partial_image, vertical_partial_image)))
                collected_partial_images.append(PartialImage(
                    image, Location(-1*horizontal_partial_image, -1*vertical_partial_image)))
                collected_partial_images.append(PartialImage(image, Location(
                    horizontal_partial_image, -1*vertical_partial_image)))
                collected_partial_images.append(PartialImage(
                    image, Location(-1*horizontal_partial_image, vertical_partial_image)))

    print('order to generate images:')
    for collected_partial_image in collected_partial_images:
        print(collected_partial_image)
    print('number of images to generate: ', len(collected_partial_images))

    # iterate through collected_partial_images and create images
    for partial_image in collected_partial_images:
        print('starting partial image for location: ', partial_image.location)
        print(
            f'{len(completed_partial_images)} images gererated of {len(collected_partial_images)}')

        above = False
        below = False
        left = False
        right = False

        if partial_image.location.x > 0:
            right = True
        elif partial_image.location.x < 0:
            left = True
        if partial_image.location.y > 0:
            above = True
        elif partial_image.location.y < 0:
            below = True

        def get_completed_partial_image_path(collected_partial_images: List[PartialImage], location: Location) -> str:
            for completed_partial_image in collected_partial_images:
                if completed_partial_image.location == location:
                    return completed_partial_image.image_path
            raise Exception(
                f'Could not find completed partial image for location {location}')

        if above and left:
            completed_below_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x, partial_image.location.y-1))
            completed_right_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x+1, partial_image.location.y))
            partial_image.image_path = create_image_above_to_left(
                below_img_path=completed_below_image_path, right_img_path=completed_right_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        elif above and right:
            completed_below_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x, partial_image.location.y-1))
            completed_left_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x-1, partial_image.location.y))
            partial_image.image_path = create_image_above_to_right(
                below_img_path=completed_below_image_path, left_img_path=completed_left_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        elif below and left:
            completed_above_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x, partial_image.location.y+1))
            completed_right_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x+1, partial_image.location.y))
            partial_image.image_path = create_image_below_to_left(
                above_img_path=completed_above_image_path, right_img_path=completed_right_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        elif below and right:
            completed_above_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x, partial_image.location.y+1))
            completed_left_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x-1, partial_image.location.y))
            partial_image.image_path = create_image_below_to_right(
                above_img_path=completed_above_image_path, left_img_path=completed_left_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        elif above:
            completed_below_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x, partial_image.location.y-1))
            partial_image.image_path = create_image_above(
                below_img_path=completed_below_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        elif below:
            completed_above_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x, partial_image.location.y+1))
            partial_image.image_path = create_image_below(
                above_img_path=completed_above_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        elif left:
            completed_right_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x+1, partial_image.location.y))
            partial_image.image_path = create_image_to_left(
                right_img_path=completed_right_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        elif right:
            completed_left_image_path = get_completed_partial_image_path(
                completed_partial_images, Location(partial_image.location.x-1, partial_image.location.y))
            partial_image.image_path = create_image_to_right(
                left_img_path=completed_left_image_path, location=partial_image.location, prompt=args.prompt, steps=args.steps)
        completed_partial_images.append(PartialImage(
            partial_image.image_path, partial_image.location))

    aggregate_images(completed_partial_images, width, height)

    def visualize_order_of_printing(max_left_right: int, max_up_down: int, collected_partial_images: List[PartialImage]):
        import matplotlib.pyplot as plt
        print("these are the collected partial images")
        plt.ion()

        above_max_left_right = max_left_right + 2
        above_max_up_down = max_up_down + 2
        plt.plot(above_max_left_right, above_max_up_down, 'ro')
        plt.plot(above_max_left_right, -1*above_max_up_down, 'ro')
        plt.plot(-1*above_max_left_right, above_max_up_down, 'ro')
        plt.plot(-1*above_max_left_right, -1*above_max_up_down, 'ro')

        plt.draw()
        input()
        for x in collected_partial_images:
            print(x)
            plt.plot(x.location.x, x.location.y, 'ro')
            plt.draw()
            input()
        print(len(collected_partial_images))
