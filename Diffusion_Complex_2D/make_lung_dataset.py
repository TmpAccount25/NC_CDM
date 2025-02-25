import os
import numpy as np
import cv2
import SimpleITK as sitk

if __name__ == "__main__":
    file_path = r"C:\Users\HLX\Desktop\recon"
    file_list = os.listdir(file_path)
    target_path = r"C:\Users\HLX\Desktop\recon_2d"
    for i in file_list:
        item_path = os.path.join(file_path,i)
        print(item_path)
        dicom = sitk.ReadImage(item_path)
        image = sitk.GetArrayFromImage(dicom)
        print(image.shape)
        # break
        for j in range(len(image)):
            H_num = image[j, :, :, 0]
            H_num = (H_num - np.min(H_num))/(np.max(H_num)-np.min(H_num))*256
            Xe_num = image[j, :, :, 1]
            Xe_num = (Xe_num - np.min(Xe_num)) / (np.max(Xe_num) - np.min(Xe_num)) * 256
            print(H_num.shape)
            cv2.imwrite(os.path.join(target_path, "real",str(i) + "_"  + str(j) + ".png"), H_num)
            print(Xe_num.shape)
            cv2.imwrite(os.path.join(target_path, "imag", str(i) + "_" + str(j) + ".png"), Xe_num)