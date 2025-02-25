import os
import SimpleITK as sitk
import numpy as np
import cv2

def create_data(file_path,target_path):
    sum_0 = 0
    sum_1 = 0
    for i in range(len(file_list)):
        item_path = os.path.join(file_path, file_list[i])
        item_list = os.listdir(item_path)

        image_path = os.path.join(item_path, item_list[4])
        label_path = os.path.join(item_path, item_list[1])
        print(image_path)
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        image_num = sitk.GetArrayFromImage(image)
        image_num = (image_num / np.max(image_num)) * 255
        label_num = sitk.GetArrayFromImage(label)
        flag = 0
        for j in range(62, 102):
            png_num = image_num[j]
            label_sum = np.sum(label_num[j])
            if label_sum == 0:
                flag = 0
                sum_0 += 1
            else:
                flag = 1
                sum_1 += 1
            cv2.imwrite(os.path.join(target_path, str(flag) + "_" + file_list[i] + str(j) + ".png"), png_num)
    print(sum_0, sum_1)

def create_label(file_path,target_path):
    sum_0 = 0
    sum_1 = 0
    for i in range(len(file_list)):
        item_path = os.path.join(file_path, file_list[i])
        item_list = os.listdir(item_path)

        image_path = os.path.join(item_path, item_list[1])
        label_path = os.path.join(item_path, item_list[1])
        print(image_path)
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        image_num = sitk.GetArrayFromImage(image)*62
        label_num = sitk.GetArrayFromImage(label)
        flag = 0
        for j in range(62, 102):
            png_num = image_num[j]
            label_sum = np.sum(label_num[j])
            if label_sum == 0:
                flag = 0
                sum_0 += 1
            else:
                flag = 1
                sum_1 += 1
            cv2.imwrite(os.path.join(target_path, str(flag) + "_" + file_list[i] + str(j) + ".png"), png_num)
    print(sum_0, sum_1)

def create_shape(file_path,target_path):
    sum_0 = 0
    sum_1 = 0
    for i in range(len(file_list)):
        item_path = os.path.join(file_path, file_list[i])
        item_list = os.listdir(item_path)

        image_path = os.path.join(item_path, item_list[0])
        label_path = os.path.join(item_path, item_list[1])
        print(image_path)
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        image_num = sitk.GetArrayFromImage(image)
        image_num[image_num!=0] = 1
        label_num = sitk.GetArrayFromImage(label)
        flag = 0
        for j in range(62, 102):
            png_num = image_num[j]
            label_sum = np.sum(label_num[j])
            if label_sum == 0:
                flag = 0
                sum_0 += 1
            else:
                flag = 1
                sum_1 += 1
            cv2.imwrite(os.path.join(target_path, str(flag) + "_" + file_list[i] + str(j) + ".png"), png_num)
    print(sum_0, sum_1)

def calculate_shape(file_path):
    max_h = 0
    min_h = 300
    # print(len(file_list[0]))
    for i in range(len(file_list)):
        max_start = 0
        min_start = 0
        item_path = os.path.join(file_path, file_list[i])
        item_list = os.listdir(item_path)

        image_path = os.path.join(item_path, item_list[0])
        image = sitk.ReadImage(image_path)
        image_num = sitk.GetArrayFromImage(image)
        image_num[image_num!=0] = 1
        for i in range(len(image_num[0])):
            if np.sum(image_num[:,i,:])==0 and max_start!=0:
                min_start = i
                break
            elif np.sum(image_num[:,i,:])!=0 and max_start==0:
                max_start = i
        # print(min_start,max_start)
        print(min_start-max_start+1)
        print(image_num.shape)

if __name__ == "__main__":
    file_path = r"E:\HLX\BraTS2020\train_all\data"
    file_list = os.listdir(file_path)
    file = r"E:\HLX\PythonProject\Diffusion_2D_BraTS\BraTS2020_2D"
    calculate_shape(file_path)
    # target_path = r"E:\HLX\PythonProject\Diffusion_2D_BraTS\BraTS2020_2D\label"
    # target_path = r"E:\HLX\PythonProject\Diffusion_2D_BraTS\BraTS2020_2D\shape"
    # create_label(file_path,target_path)
    # create_shape(file_path,target_path)