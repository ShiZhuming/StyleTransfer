# import requests as req
from PIL import Image
from io import BytesIO

# 制作缩略图
def make_thumb(filename, sizes=(128, 128)):
    f = open(filename,'rb')
    try:
        im = Image.open(f)
    except BaseException:
        f.close()
        return
    mode = im.mode
    if mode not in ('L', 'RGB'):
        if mode == 'RGBA':
            # 透明图片需要加白色底
            alpha = im.split()[3]
            bgmask = alpha.point(lambda x: 255 - x)
            im = im.convert('RGB')
            im.paste((255, 255, 255), None, bgmask)
        else:
            im = im.convert('RGB')

    # 切成方图，避免变形
    width, height = im.size
    if width == height:
        region = im
    else:
        if width > height:
            # h*h
            delta = (width - height) / 2
            box = (delta, 0, delta + height, height)
        else:
            # w*w
            delta = (height - width) / 2
            box = (0, delta, width, delta + width)
        region = im.crop(box)

    # resize
    thumb = region.resize((sizes[0], sizes[1]), Image.ANTIALIAS)
    f.close()
    #保存图片
    thumb.save(filename.replace('upload','thumb'), quality=100)

# from os import listdir

# if __name__ == '__main__':
#     for p in ['convert/']:
#         for f in listdir('image/upload/'+p):
#             if f != 'README.md':
#                 make_thumb('image/upload/'+p+f,sizes=(256,256))
#     for p in ['content/','style/']:
#         for f in listdir('image/upload/'+p):
#             if f != 'README.md':
#                 make_thumb('image/upload/'+p+f)
