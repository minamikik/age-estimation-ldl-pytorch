import os
import requests
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from facenet_pytorch import MTCNN

from .age_model import get_model
from .age_dataset import expand_bbox
from .age_defaults import _C as config

class AgePredictor:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path

        try:
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = 'https://github.com/yjl450/age-estimation-ldl-pytorch/releases/download/v1.0/megaage_fusion.pth'
                r = requests.get(url, allow_redirects=True)
                open(model_path, 'wb').write(r.content)

            self.device = device

            self.model = get_model(model_name=config.MODEL.ARCH, pretrained=None)

            self.model.eval()
            self.model = self.model.to(self.device)

            checkpoint = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(checkpoint['state_dict'])

            self.mtcnn = MTCNN(device=self.device, post_process=False, keep_all=False)
            self.img_size = config.MODEL.IMG_SIZE
        except Exception as e:
            raise e


    def predict(self, pil_img):
        try:
            predicted_ages=[0]
            detected=[[0,0]]

            with torch.no_grad():
                image = pil_img
                detected, _, landmarks = self.mtcnn.detect(image, landmarks=True)

                if detected is not None and len(detected) > 0:
                    detected = detected.astype(int)
                    box = detected[0]
                    image = image.crop(box)
                    # image.show()
                    image.resize((self.img_size,self.img_size))
                    image = torchvision.transforms.ToTensor()(image)
                    image = image.unsqueeze(0).to(self.device)

                    # predict ages
                    outputs = self.model(image)
                    outputs = F.softmax(outputs, dim=1)
                    _, predicted_ages = outputs.max(1)

                    age = predicted_ages[0]

                else:
                    age = -1

            torch.cuda.empty_cache()
            return age
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

