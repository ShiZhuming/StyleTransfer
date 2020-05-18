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
import torch
import argparse
import torchvision.transforms as transforms
from net import FPnet,Decoder
from PIL import Image, ImageFile

from torchvision.utils import save_image



####################options###################
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-c','--content_img',type=str, required=True,
                    help='path of your content image')
parser.add_argument('-s','--style_img',type=str, required=True,
                    help='path of your style image')
parser.add_argument('-m','--model_path', default='model/decoder.pth',
                    help='path of the trained model')
parser.add_argument('--lamda', type=float, default=10.0)
parser.add_argument('--alpha', type=float, default=1.0)

parser.add_argument('--save_dir', default='result/result.jpg',
                    help='Directory to save the result image')

args = parser.parse_args()

def test(contentpath,stylepath):
    '''一次前传得到风格化图像'''
    contentimg = Image.open(str(contentpath)).convert('RGB')
    styleimg = Image.open(str(stylepath)).convert('RGB')
    contentimg=transforms.Compose([transforms.ToTensor()])(contentimg)
    styleimg=transforms.Compose([transforms.ToTensor()])(styleimg)


    contentimg=contentimg.view(1,3,contentimg.shape[1],contentimg.shape[2])
    styleimg=styleimg.view(1,3,styleimg.shape[1],styleimg.shape[2])



    decoder=Decoder()
    decoder.load_state_dict(torch.load(args.model_path))

    fbnet=FPnet(decoder).cuda()
    output=fbnet(contentimg,styleimg,alpha=args.alpha,lamda=args.lamda,require_loss=False)


    save_image(output.cpu(),args.save_dir)
    print('image saved.')


if __name__ == "__main__":
    test(args.content_img,args.style_img)