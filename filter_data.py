import os
import glob 
from collections import Counter

path1 = glob.glob("/data/disk1/hungpham/NudeNet/data/train/images/*")
A =  set()
B = set()
for i in path1:
    i = str(i)
    # print(i)
    # if os.path.exists(f"")
    if i[-3:] == "jpg":
        a = i.split("/")[-1][:-4]
    else:
        a = i.split("/")[-1][:-5]
    A.add(a)

path2 = glob.glob("/data/disk1/hungpham/NudeNet/data/train/labels/*")
for i in path2:
    i = str(i)
    b = i.split("/")[-1][:-4]
    B.add(b)
    
print(len(A))
print(len(B))

C = B.difference(A)
print(len(C))

for i in C:
    os.system("rm -rf /data/disk1/hungpham/NudeNet/data/train/labels/{}..txt".format(i))

# _, _, files = next(os.walk("/home/data/hungpham/Training/images1"))
# file_count = len(files)

# print(file_count)
# # print(C)