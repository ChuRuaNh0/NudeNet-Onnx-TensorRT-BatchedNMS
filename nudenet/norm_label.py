import glob
import os
import cv2

if __name__ == "__main__":
    for path_txt in glob.glob("/data/disk1/hungpham/NudeNet/data/train/labels/*.txt"):
        path_img1 = path_txt[:-3].replace("/labels/", "/images/") + "jpg"
        path_img2 = path_txt[:-3].replace("/labels/", "/images/") + "png"
        path_img3 = path_txt[:-3].replace("/labels/", "/images/") + "jpeg"
        if os.path.exists(path_img1):
            path_img = path_img1
        elif os.path.exists(path_img2):
            path_img = path_img2
        else:
            path_img = path_img3
        file = open(path_txt, "r")
        file.read().splitlines()
       