# encoding:utf-8
import os
import numpy as np
import glob2
import os
from PIL import Image
import matplotlib.pyplot as plt

def png2jpg(path):
    img_png = Image.open(path)
    print(img_png.mode, img_png.size)
    plt.imshow(img_png)
    img_pil = img_png.convert('RGBA')
    x, y = img_pil.size
    img_jpg = Image.new('RGBA', img_pil.size, (0, 255, 0))
    img_jpg.paste(img_pil, (0, 0, x, y), img_pil)
    img_jpg = img_jpg.convert("RGB")
    print(img_jpg.mode, img_jpg.size)
    plt.imshow(img_jpg)
    img_jpg.save(path.split('.')[0]+'.jpg')

if __name__ == "__main__":

    basePath = "D:\\Projects\\render_kit\\obj_render_blender\\temp\\renderresult\\2\\shade"
    
    path_vec = glob2.glob(r"{0}/*.png".format(basePath))
    count = 0
    for obj_path in path_vec:
        count += 1
        png2jpg(obj_path)
        print("obj_path:{0}|count:{1}|".format(obj_path, count))





