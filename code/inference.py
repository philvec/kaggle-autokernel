import sys
import models
import torch

PROJECT_NAME = sys.environ('PROJECT_NAME')
sys.path.insert(1, f'/kaggle/input/{PROJECT_NAME}/code')

from dataloaders import load_dcm_as_tensor


class ProjectInferrer:
    def __init__(self, images_path, model_path):
        self.images_path = images_path
        self.model = torch.load(model_path)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, x):
        # pred = self.softmax(self.model(x))
        pred = self.model(x)
        return pred
