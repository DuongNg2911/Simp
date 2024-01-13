import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from .model import load_model

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def output_to_img(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out 

@torch.inference_mode()
def infer_deepfill(generator, image, mask, return_vals=['inpainted', 'stage1']):
    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    image = (image * 2 - 1.)
    mask = (mask > 0.).to(dtype=torch.float32)
    image_masked = image * (1. - mask)

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask], dim=1)
    x_stage1, x_stage2 = generator(x, mask)

    image_compl = image * (1. - mask) + x_stage2 * mask
    output = []

    for return_val in return_vals:
        if return_val.lower() == 'stage1':
            output.append(output_to_img(x_stage1))
        elif return_val.lower() == 'stage2':
            output.append(output_to_img(x_stage2))
        elif return_val.lower() == 'inpainted':
            output.append(output_to_img(image_compl))
        else:
            print(f'Invalid return value: {return_val}')
    
    return output

class Inpainter:
    def __init__(self, config_p, device='cpu') -> None:
        with open(config_p, 'r') as stream:
            self.config = yaml.load(stream, Loader)
        if self.config['tf_places2']['load_at_startup']:
            self.model = load_model(self.config['tf_places2']['path'], device)

        self.device = device

    def inpaint(self, image_p, mask_p, max_size=512):
        mw, mh = mask_p.size
        scale = max_size/max(mw, mh)

        mask_pil = mask_p.resize((max_size, int(scale*mh)) if mw > mh else (int(scale*mw), max_size))
        image_pil = image_p.resize(mask_pil.size)

        image, mask = ToTensor()(image_pil), ToTensor()(mask_pil)
        return_vals = self.config['tf_places2']['return_vals']
        outputs = infer_deepfill(self.model, image.to(self.device), mask.to(self.device), return_vals=return_vals)

        return outputs[0]