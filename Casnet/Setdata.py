import os
from torch.utils.data import Dataset
import numpy as np
import h5py
from os.path import splitext
from os import listdir
import logging
import pickle
import torch
import matplotlib.pyplot as plt
import SimpleITK as itk
from skimage import transform
import scipy.io as scio
import pydicom

# 构建自己的训练集，样本和标签
class MyDataset(Dataset):#继承了Dataset子类
    # def __init__(self,input_root,label_root,transform=None):
    #     #分别读取输入/标签图片的路径信息
    #     self.input_root=input_root
    #     self.input_files=os.listdir(input_root)#列出指定路径下的所有文件
    #     self.label_root=label_root
    #     self.label_files=os.listdir(label_root)
    #     self.transforms=transform
    # def __len__(self):
    #     #获取数据集大小
    #     return len(self.input_files)
    # def __getitem__(self, index):
    #     #根据索引(id)读取对应的图片
    #     input_img_path=os.path.join(self.input_root,self.input_files[index])
    #     label_img_path=os.path.join(self.label_root,self.label_files[index])
    #     input_img=np.load(input_img_path)
    #     label_img=np.load(label_img_path)
    #     return input_img,label_img    #返回成对的数据
    def __init__(self,input_root,transform=None):
        self.file_path = input_root
        self.kspace_train = h5py.File(input_root, "r")['img'][:]
        self.transforms=transform
    def __len__(self):
        return len(self.kspace_train)
    def __getitem__(self,index):
        input_img = list(self.kspace_train)[index-1]
        return input_img
    
    # def __init__(self):
    #     self.file_path = './data/faces/'
    #     f=open("final_train_tag_dict.txt","r")
    #     self.label_dict=eval(f.read())
    #     f.close()

    # def __getitem__(self,index):
    #     label = list(self.label_dict.values())[index-1]
    #     img_id = list(self.label_dict.keys())[index-1]
    #     img_path = self.file_path+str(img_id)+".jpg"
    #     img = np.array(Image.open(img_path))
    #     return img,label

    # def __len__(self):
    #     return len(self.label_dict)


class IXIdataset(Dataset):
    def __init__(self, data_dir,validtion_flag=False):
        self.data_dir = data_dir
        self.validtion_flag = validtion_flag

        self.img_size = [256,150]

        #make an image id's list
        self.file_names = [splitext(file)[0] for file in listdir(data_dir)
                    if not file.startswith('.')]

        self.ids = list()
        slice_range = [90,190]
        for file_name in self.file_names:
            try:
                # full_file_path = path.join(self.data_dir,file_name+'.hdf5')
                full_file_path = self.data_dir + '/' + file_name + '.h5'
                with h5py.File(full_file_path, 'r') as f:
                    numOfSlice = f['data'].shape[0]

                if numOfSlice < slice_range[1]:
                    continue

                for slice in range(slice_range[0], slice_range[1]):
                    self.ids.append((file_name, slice))
            except:
                continue

        if self.validtion_flag:
            logging.info(f'Creating validation dataset with {len(self.ids)} examples')
        else:
            logging.info(f'Creating training dataset with {len(self.ids)} examples')


        #random noise:  minmax_noise_val: [-0.01, 0.01]
        # self.minmax_noise_val = args.minmax_noise_val

    def __len__(self):
        return len(self.ids)

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[0] == self.img_size[0]:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size[0])/2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]
        # return myFFT2.k_ifft2_image_np(kspace_cplx)[None, :, :]

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))
        # return myFFT2.image_fft2_k_np(img)

    # @classmethod
    def slice_preprocess(self, kspace_cplx):
        #crop to fix size
        kspace_cplx = self.crop_toshape(kspace_cplx)
        #split to real and imaginary channels
        kspace = np.zeros((self.img_size[0], self.img_size[1], 2))
        #target image:

        # noise_real = np.random.normal(0, 10, size=kspace_cplx.shape)
        # noise_imag = np.random.normal(0, 10, size=kspace_cplx.shape)

        # kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32) + noise_real
        # kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32) + noise_imag
        
        # kspace_cplx = kspace[:,:,0] + 1j*kspace[:,:,1]
        noise = np.load('noise.npy')
        kspace_cplx = kspace_cplx + noise

        image = self.ifft2(kspace_cplx)

        kspace_cplx=self.fft2(image)
        kspace_cplx=self.fft2(image)
        # HWC to CHW
        # kspace_cplx = kspace_cplx.transpose((2, 0, 1))
        kspace_cplx = kspace_cplx[0,:,:]

        # masked_Kspace += np.random.uniform(low=self.minmax_noise_val[0], high=self.minmax_noise_val[1],
        #                                    size=masked_Kspace.shape)*self.maskedNot
        return kspace_cplx, image

    def __getitem__(self, i):
        file_name, slice_num = self.ids[i]

        full_file_path = self.data_dir + '/' + file_name + '.h5'

        with h5py.File(full_file_path, 'r') as f:
            imgs = f['data'][:,slice_num, :]

        target_Kspace = np.zeros((2, self.img_size[0], self.img_size[1]))
        target_img = np.zeros((1, self.img_size[0], self.img_size[1]))

        
        img = imgs
        img = img/np.max(np.abs(img))
        kspace = self.fft2(img)
        slice_full_Kspace, slice_full_img = self.slice_preprocess(kspace)

        # plt.subplot(121)
        # plt.imshow(np.abs(slice_full_img[0]),'gray')
        # plt.show()



        target_Kspace = slice_full_Kspace
        target_img = slice_full_img

        return torch.from_numpy(target_Kspace),torch.from_numpy(target_img)
    

class SliceData(Dataset):
    def __init__(self, root, transform, batch_size):
        self.transform = transform
        self.example = []
        files = os.listdir(root)
        for file in files:
            fname = os.path.join(root, file)
            # data = np.load(fname)
            matdata = scio.loadmat(fname)['img_xe']
            # print(data.shape, fname)
            if len(matdata.shape) == 3:
                data = np.transpose(matdata, (2, 0, 1))
                # data = np.fft.fftshift(np.transpose(matdata, (2, 0, 1)))
                num_slices = data.shape[0]
                self.example += [(fname, slice) for slice in range((num_slices // batch_size) * batch_size)]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        fname, slice = self.example[i]
        data = np.transpose(scio.loadmat(fname)['img_xe'], (2, 0, 1))
        # data = np.fft.fftshift(np.transpose(scio.loadmat(fname)['k_xe'], (2, 0, 1)))
        kspace = data[slice]
        return self.transform(kspace,slice)
    

class ThreeD_Data(Dataset):
    def __init__(self, root, transform, batch_size):
        self.transform = transform
        self.example = []
        files = os.listdir(root)
        for file in files:
            fname = os.path.join(root, file)
            # data = np.load(fname)
            matdata = scio.loadmat(fname)['img_xe']
            # print(data.shape, fname)
            if len(matdata.shape) == 3:
                data = np.transpose(matdata, (2, 0, 1))
                self.example += [fname]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        fname = self.example[i]
        data = np.transpose(scio.loadmat(fname)['img_xe'], (2, 0, 1))
        image = data
        return self.transform(image)
    
class IMA_Data(Dataset):
    def __init__(self, root, transform, batch_size):
        self.transform = transform
        self.example = []
        files = os.listdir(root)
        for file in files:
            fname = os.path.join(root, file)
            imadata = pydicom.read_file(fname)
            img = imadata.pixel_array
            # plt.imshow(img,'gray')
            # plt.show()
            # print(data.shape, fname)
            self.example += [fname]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        fname = self.example[i]
        data = pydicom.read_file(fname)
        image = data.pixel_array
        return self.transform(image)
    
class Nii_Data(Dataset):
    def __init__(self, root, transform, batch_size):
        self.transform = transform
        self.example = []
        files = os.listdir(root)
        for file in files:
            fname = os.path.join(root, file)
            # data = np.load(fname)
            imgdata = itk.ReadImage(fname)
            data = itk.GetArrayFromImage(imgdata)
            # plt.subplot(133)
            # plt.imshow(data[0,:,:],'gray')
            # plt.show()
            num_slices = data.shape[0]
            self.example += [(fname, slice) for slice in range((num_slices // batch_size) * batch_size)]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        fname, slice = self.example[i]
        # data = np.load(fname)
        imgdata = itk.ReadImage(fname)
        data = itk.GetArrayFromImage(imgdata)
        # data = transform.resize(data, (data.shape[0], 256, 150))
        # ks = np.fft.fft2(data)
        # kspace = ks[slice]
        img = data[slice]
        return self.transform(img, slice)