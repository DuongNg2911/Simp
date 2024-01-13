import logging
import numpy as np
import onnxruntime as ort

from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx
)

from PIL.Image import Image as PILImage
from PIL import Image
from typing import Dict, List, Tuple

class U2netpONNX:
    "Model to remove background automatically"

    def __init__(self, model_path) -> None:
        providers = ort.get_available_providers()
        self.target_size = (320, 320)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        providers = [p for p in providers if p != "TensorrtExecutionProvider"]
        if providers:
            logging.info(
                "Available providers for ONNXRuntime: %s", ", ".join(providers)
            )
        else:
            logging.warning("No available providers for ONNXRuntime")

        self.session = ort.InferenceSession(
            model_path, providers=providers
        )
    
    def preprocessing(self, 
                    img: PILImage):
        im = img.convert("RGB").resize(self.target_size, Image.LANCZOS)

        im_array = np.array(im)
        im_array = im_array / np.max(im_array)

        tempImg = np.zeros((im_array.shape[0], im_array.shape[1], 3))
        tempImg[:,:,0] = (im_array[:,:,0] - self.mean[0])/self.std[0]
        tempImg[:,:,1] = (im_array[:,:,1] - self.mean[1])/self.std[1]
        tempImg[:,:,2] = (im_array[:,:,2] - self.mean[2])/self.std[2]

        tempImg = tempImg.transpose((2,0,1))
        
        return {
            self.session.get_inputs()[0].name:
            np.expand_dims(tempImg, 0).astype(np.float32)
        }

    def predict(self, dict_img: Dict, img_size: Tuple) -> List[PILImage]:
        ort_outs = self.session.run(
            None,
            dict_img
        )

        pred = ort_outs[0][:,0,:,:]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred-mi)/(ma-mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred*255).astype("uint8"), mode="L")
        mask = mask.resize(img_size, Image.LANCZOS)

        return [mask]
    
    def postprocessing(self, mask: np.ndarray) -> np.ndarray:
        """
        Post Process the mask for a smooth boundary by applying Morphological Operations
        Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
        """
        kernel = getStructuringElement(MORPH_ELLIPSE, (3,3))

        mask = morphologyEx(mask, MORPH_OPEN, kernel)
        mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
        mask = np.where(mask < 127, 0, 255).astype(np.uint8)

        return mask