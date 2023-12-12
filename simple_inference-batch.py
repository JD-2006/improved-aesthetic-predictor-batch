import os
from PIL import Image
import torch
from torch.nn import functional as F
from warnings import filterwarnings
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from os.path import join
import json
import clip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose GPU if you are on a multi-GPU server

# Specify the folder containing images
img_folder = r"C:\Users\CHP_7575\Documents\Image_Scripts\improved-aesthetic-predictor-batch"

# Get a list of image paths in the folder
img_paths = [os.path.join(img_folder, img_name) for img_name in os.listdir(img_folder) if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Create an empty list to store results
results = []

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# Load the MLP model
model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
model_state_dict = torch.load("sac+logos+ava1-l14-linearMSE.pth")  # load the model you trained previously or the model available in this repo
model.load_state_dict(model_state_dict)

model.to("cuda")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

# Iterate through images
for img_path in tqdm(img_paths, desc="Processing images"):
    pil_image = Image.open(img_path)
    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy())

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    # Append the result (score, file_name) to the list
    results.append((prediction.item(), os.path.basename(img_path)))

# Sort the results in descending order based on the aesthetic score
results.sort(reverse=True)

# Print the results
print("Aesthetic scores in descending order:")
for score, file_name in results:
    print(f"{file_name}: {score}")
