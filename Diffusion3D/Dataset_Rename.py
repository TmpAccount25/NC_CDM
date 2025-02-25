import os

if __name__ == "__main__":
    root = r"F:\BraTS2024\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData"
    item_list = os.listdir(root)
    for i in range(len(item_list)):
        item_path = os.path.join(root,item_list[i])
        image_list = os.listdir(item_path)
        print(image_list)
        print(item_path)
        print(os.path.join(item_path,image_list[2]))
        # print(os.path.join(item_path,image_list[2].replace("t1n","t1")))
        os.rename(os.path.join(item_path,image_list[2]),os.path.join(item_path,image_list[2]).replace("t1n","t1"))
        os.rename(os.path.join(item_path,image_list[3]),os.path.join(item_path,image_list[3]).replace("t2f","flair"))
        os.rename(os.path.join(item_path,image_list[4]),os.path.join(item_path,image_list[4]).replace("t2w","t2"))