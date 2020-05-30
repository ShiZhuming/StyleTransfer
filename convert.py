import os
import cv2
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from net import FPnet,Decoder
from PIL import Image, ImageFile
from function import coral,change_color
from torchvision.utils import save_image

def test_transform(size):
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def transfer(contentpath,stylepath,converted,pixel=512,model_path='static/20200522decoder100000_1.pth'):
    '''一次前传得到风格化图像'''
    mytransfer=test_transform(pixel)

    contentfile = open(str(contentpath),'rb')
    stylefile = open(str(stylepath),'rb')

    contentimg = Image.open(contentfile).convert('RGB')
    styleimg = Image.open(stylefile).convert('RGB')
    
    contentfile.close()
    stylefile.close()

    contentimg=mytransfer(contentimg).unsqueeze(0)
    styleimg=mytransfer(styleimg).unsqueeze(0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    decoder=Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # decoder = decoder.module
    # decoder.load_state_dict(torch.load(model_path))

    fbnet=FPnet(decoder,True).to(device).eval()
    output=fbnet(contentimg,styleimg,alpha=1.0,lamda=1.0,require_loss=False)

    save_image(output.cpu(),converted)
    contentimg.detach()
    styleimg.detach()




                

        
