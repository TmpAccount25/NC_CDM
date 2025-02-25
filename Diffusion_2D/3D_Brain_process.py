import SimpleITK as sitk
import os
import shutil
import numpy as np

#136,184,144
def cal_end(num):
    start_x = 0
    end_x = 0
    for i in range(len(num)):
        if np.sum(num[i,:,:]) == 0 and start_x!=0:
            end_x = i
            break
        elif np.sum(num[i,:,:]) != 0 and start_x==0:
            start_x = i
    x = (end_x+start_x)//2
    if x < 68:
        x =0
    else:
        x-=68
    print(x)
    print(end_x, start_x)

    start_y = 0
    end_y = 0
    for i in range(len(num[0])):
        if np.sum(num[:, i, :]) == 0 and start_y != 0:
            end_y = i
            break
        elif np.sum(num[:, i, :]) != 0 and start_y == 0:
            start_y = i
    y = (end_y + start_y) // 2
    print(end_y,start_y)
    if y < 92:
        y = 0
    else:
        y -= 92
    print(y)

    start_z = 0
    end_z = 0
    for i in range(len(num[0][0])):
        if np.sum(num[ :, :,i]) == 0 and start_z != 0:
            end_z = i
            break
        elif np.sum(num[ :, :,i]) != 0 and start_z == 0:
            start_z = i
    z = (end_z + start_z) // 2
    if z < 72:
        z = 0
    else:
        z -= 72
    print(z)
    print(end_z, start_z)
    return x,y,z

if __name__ == "__main__":
    all_path = r"C:\Users\HLX\Desktop\BraTS2020"
    path_list = os.listdir(all_path)
    for i in range(len(path_list)):
        item_path = path_list[i]
        item_list = os.listdir(os.path.join(all_path,item_path))

        flair_path = os.path.join(all_path,path_list[i],item_list[0])
        flair_image = sitk.ReadImage(flair_path)
        flair_num = sitk.GetArrayFromImage(flair_image)
        x,y,z = cal_end(flair_num)
        flair_num = flair_num[x:x+136,y:y+184,z:z+144]
        flair_image = sitk.GetImageFromArray(flair_num)

        # print(flair_num.shape)

        seg_path = os.path.join(all_path, path_list[i], item_list[1])
        seg_image = sitk.ReadImage(seg_path)
        seg_num = sitk.GetArrayFromImage(seg_image)
        seg_num = seg_num[x:x + 136, y:y + 184, z:z + 144]
        seg_image = sitk.GetImageFromArray(seg_num)

        # print(seg_num.shape)

        t1_path = os.path.join(all_path, path_list[i], item_list[2])
        t1_image = sitk.ReadImage(t1_path)
        t1_num = sitk.GetArrayFromImage(t1_image)
        t1_num = t1_num[x:x + 136, y:y + 184, z:z + 144]
        t1_image = sitk.GetImageFromArray(t1_num)

        # print(t1_num.shape)

        t1ce_path = os.path.join(all_path, path_list[i], item_list[3])
        t1ce_image = sitk.ReadImage(t1ce_path)
        t1ce_num = sitk.GetArrayFromImage(t1ce_image)
        t1ce_num = t1ce_num[x:x + 136, y:y + 184, z:z + 144]
        t1ce_image = sitk.GetImageFromArray(t1ce_num)

        # print(t1ce_num.shape)

        t2_path = os.path.join(all_path, path_list[i], item_list[4])
        t2_image = sitk.ReadImage(t2_path)
        t2_num = sitk.GetArrayFromImage(t2_image)
        t2_num = t2_num[x:x + 136, y:y + 184, z:z + 144]
        t2_image = sitk.GetImageFromArray(t2_num)
        sitk.WriteImage(flair_image, flair_path)
        sitk.WriteImage(seg_image, seg_path)
        sitk.WriteImage(t1_image, t1_path)
        sitk.WriteImage(t1ce_image, t1ce_path)
        sitk.WriteImage(t2_image, t2_path)
        # print(t2_num.shape)
        if t2_num.shape!=(136,184,144):
            print(t2_num.shape)
            print(item_path)