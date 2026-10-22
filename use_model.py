import torch

import torch.optim as optim
from torchvision import transforms

import argparse

from PIL import Image

from utils import load_checkpoint
from models.baseline_net import BaselineNet
from models.unet import UNet
from models.DerainNet import DerainNet
from models.Mymodel import Mymodel


parser = argparse.ArgumentParser(description="Deraining Model Training")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data")
parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet","derainnet","mymodel"], help="Model to use")
parser.add_argument("--checkpoint", required=True,type=str, default=None, help="Path to checkpoint for testing")
args=parser.parse_args()


if args.model=="unet":
    model=UNet()
elif args.model=="derainnet":
    model=DerainNet()
elif args.model=="mymodel":
    model=Mymodel()
elif args.model=="baseline":
    model=BaselineNet()

data=args.data_dir
checkpoint=args.checkpoint
device="cuda" if torch.cuda.is_available() else "cpu"
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model=model.to(device).eval()


def load_img(Path):
    device="cuda" if torch.cuda.is_available() else "cpu"
    img = Image.open(Path).convert("RGB")
    
    transform= transforms.Compose([
                
                transforms.ToTensor()])
    img=transform(img)
    img=img.unsqueeze(0).to(device)
    return img

load_checkpoint(torch.load(checkpoint, map_location=device), model, optimizer)
img_in=load_img(data)


with torch.no_grad():
    img=model(img_in)
    final_img = img.squeeze(0).clamp(0, 1).cpu().detach()
    final_img = transforms.ToPILImage()(final_img)
    final_img.save(f"output_{args.model}.jpg")


