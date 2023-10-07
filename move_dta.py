import glob
import os
from tqdm import tqdm

if __name__ == "__main__":
    path_txt = "/data/disk1/hungpham/NudeNet/nudenet/save_txt/*.txt"
    path_img = "/data/disk1/vinhnguyen/resnet_uniform/training/nude"
    for path_txt in tqdm(glob.glob(path_txt)):
        name_txt = path_txt.split("/")[-1][:-4]
        name_img1 = name_txt + ".jpg"
        name_img2 = name_txt + ".jpeg"
        # print(name_img1)
        # print(name_img2)
        # exit(0)
        if os.path.exists(f"{path_img}/{name_img1}"):
            os.system(f"cp {path_img}/{name_img1} /data/disk1/hungpham/NudeNet/data/train/images/")
        else:
            os.system(f"cp {path_img}/{name_img2} /data/disk1/hungpham/NudeNet/data/train/images/")