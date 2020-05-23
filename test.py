'''
stylise the image within one propagation

Run the script with:
```
python test.py -c path_to_your_content_image.jpg \
    -s path_to_your_style_image.jpg
```
e.g.:
```
python test.py -c content/golden_gate.jpg -s style/la_muse.jpg
```

Optional parameters:
```
-m, Path of the trained model (Default is "model/decoder.pth")
--save_dir, Directory to save the result image.jpg  (Default is "result/result.jpg")
```

'''


import os
import cv2
import glob
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from net import FPnet,Decoder
from PIL import Image, ImageFile
from function import coral,change_color
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################options###################
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-c','--content',type=str, 
                    help='path of your content image')
parser.add_argument('-s','--style',type=str, 
                    help='path of your style image')
parser.add_argument('-cd','--content_dir',type=str, 
                    help='path of your style image')
parser.add_argument('-sd','--style_dir',type=str, 
                    help='path of your style image')
parser.add_argument('-m','--model_path', default='model/decoder.pth',
                    help='path of the trained model')
parser.add_argument('--lamda', type=float, default=10.0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('-p','--preserve_color', type=bool, default=False,
                   help='preserve the color of the content image')
parser.add_argument('--save_dir', default='result',
                    help='Directory to save the result image')

args = parser.parse_args()



def test_transform(size):
    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform








def test(contentpath,stylepath,multi=False):
    '''一次前传得到风格化图像'''
    if multi==False:
        content_name=(contentpath.split('/')[-1]).split('.')[0]
        style_name=(stylepath.split('/')[-1]).split('.')[0]
    else:
        content_name=(contentpath.split('\\')[-1]).split('.')[0]
        style_name=(stylepath.split('\\')[-1]).split('.')[0]

    transfer=test_transform(512)

    contentimg = Image.open(str(contentpath)).convert('RGB')
    styleimg = Image.open(str(stylepath)).convert('RGB')
    if args.preserve_color: styleimg = change_color(styleimg, contentimg)
    contentimg=transfer(contentimg).unsqueeze(0)
    styleimg=transfer(styleimg).unsqueeze(0)

    #if args.preserve_color: styleimg = coral(styleimg, contentimg)
    

    decoder=Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(args.model_path))

    fbnet=FPnet(decoder,True).to(device).eval()
    output=fbnet(contentimg,styleimg,alpha=args.alpha,lamda=args.lamda,require_loss=False)

    image_name=args.save_dir+'/'+content_name+'+'+style_name+'.jpg'
    save_image(output.cpu(),image_name)
    print('image saved  as:  '+image_name)
    contentimg.detach()
    styleimg.detach()


if __name__ == "__main__":
    assert (args.content or args.content_dir)
    assert (args.style or args.style_dir)
    if args.content and args.style:
        test(args.content,args.style)
    elif args.content_dir and args.style_dir:
        contents=glob.glob(args.content_dir+"/*.jpg")
        styles=glob.glob(args.style_dir+"/*.jpg")
        for content in contents:
            for style in styles:
                test(content,style,multi=True)
    
    print('COMPLETED.')
                

        
