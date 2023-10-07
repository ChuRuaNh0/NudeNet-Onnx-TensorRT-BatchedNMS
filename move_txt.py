import glob
import os
path2 = glob.glob("/data/disk1/hungpham/NudeNet/nudenet/save_txt/*")
for i in path2:
    os.system(f"cp {i} /data/disk1/hungpham/NudeNet/data/train/labels/")