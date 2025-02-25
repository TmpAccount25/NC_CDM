import SimpleITK as sitk
import os
import numpy as np

if __name__ == "__main__":
    path = r"C:\Users\HLX\Desktop\generation_lung_all"
    path_list = os.listdir(path)
    total_image = np.zeros((1,96,96))
    for i in range(len(path_list)):
        if ".nii.gz" not in path_list[i]:
            # print(path_list[i])
            continue
        image = sitk.ReadImage(os.path.join(path,path_list[i]))
        image_num = sitk.GetArrayFromImage(image)
        image_num = (image_num[0:1]+1)/2
        # print(image_num.shape)
        total_image = np.concatenate((total_image,image_num),axis=0)
        print(total_image.shape)
    total_image = total_image[1:]
    image = sitk.GetImageFromArray(total_image)
    sitk.WriteImage(image,"train_gen.nii.gz")