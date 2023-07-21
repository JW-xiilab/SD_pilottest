import os
import torch
import logging
import argparse
import requests
from io import BytesIO
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpaining Pilot-test")

    parser.add_argument("--id", default="1")
    parser.add_argument("--root_dir", default="/DATA_17/kjw/SD_pilottest")
    parser.add_argument('--output_path', default='output')
    parser.add_argument("--log_output", default='logs')
    args = parser.parse_args()
    return args

def main(args):
    # Setup prerequisites
    output_path = os.path.join(args.root_dir, args.output_path, args.id)
    root_dir = os.path.join(args.root_dir, 'input_img')
    os.makedirs(os.path.join(root_dir, args.log_output), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    out_dirs = ['fire', 'hole','leak','destruction','crack','cutoff']
    for out_d in out_dirs:
        os.makedirs(os.path.join(output_path, out_d), exist_ok=True)
    img_paths = os.listdir(os.path.join(root_dir, 'crop'))
    img_paths = sorted(img_paths, key=lambda x: int(os.path.splitext(x)[0]))

    # Setup logger
    logger = logging.getLogger(name='Ext.Log')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(f'{os.path.join(output_path, args.log_output)}.log', mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )

    # Setup inference model
    pipeline = pipeline.to("cuda")
    logger.info(f'Training Starts')
    texts = ["Draw a picture of a big fire with smoke.",
             "Draw a picture of a building with a sink hole in it",
             "Draw a picture of a liquid leaking",
             "This image is an aerial view. Draw me a picture of the roof of \
                  a building completely destroyed.",
             "Draw me a picture of a building full of cracks.",
             "Draw a picture of a building that has been completely cut off"]
    
    for img_p in img_paths:
        img = Image.open(os.path.join(root_dir, 'crop', img_p))
        mask_img = Image.open(os.path.join(root_dir, 'mask', img_p))
        img = img.resize((512,512))
        mask_img = mask_img.resize((512,512))
        image = pipeline(prompt=texts, image=img, mask_image=mask_img)#.images[0]
        logger.info(f'Image saving')
        for i in range(len(texts)):
            out_img = image.images[i]
            out_img.save(os.path.join(output_path, out_dirs[i], img_p))



if __name__ == "__main__":
    args = parse_args()
    main(args)
