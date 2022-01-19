import glob

im_list = glob.glob("yourpath/JPEGImages/*")

f_trainval = open("yourpath/ImageSets/Main/trainval.txt", "w")

f_test = open("yourpath/ImageSets/Main/test.txt", "w")

for idx in range(im_list.__len__()):
    im_path = im_list[idx]
    print(im_path)
    if idx < 0.9 * im_list.__len__():
        f_trainval.writelines(im_path.split('.')[0].split("/")[-1] + "\n")
    else:
        f_test.writelines(im_path.split('.')[0].split("/")[-1] + "\n")

f_test.close()
f_trainval.close()
