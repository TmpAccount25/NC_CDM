import argparse
import shutil
import time
import pathlib
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils import compressed_sensing as cs
from skimage.metrics import structural_similarity as ssim
from tensorboardX import SummaryWriter
from utils import metric
from tqdm import tqdm
from CasNet import UNetModel
import SimpleITK as itk
import myFFT2
import os
from torch.utils.data import Dataset
import scipy.io as scio

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
        # 添加一个Mask计算指标
        imgdata = itk.ReadImage('lung_mask.nii')
        lung_mask = itk.GetArrayFromImage(imgdata)[slice]
        img = data[slice]
        # plt.subplot(121)
        # plt.imshow(np.abs(lung_mask),'gray')
        # plt.subplot(122)
        # plt.imshow(np.abs(img),'gray')
        # plt.show()
        return self.transform(img, slice),lung_mask

class DataTransform:
    def __call__(self, kspace, slice):
        target_kspace = kspace
        target_kspace_tensor = torch.from_numpy(target_kspace).float()
        return target_kspace_tensor

def prep_input(x):

    '输入全采样图像'
    shape = x.shape
    mask = torch.zeros(x.shape).to(args.device)
    # 随机产生欠采样矩阵
    # Mask = np.rot90(cs.cartesian_mask((shape[2],shape[1]), acc=4, sample_n=8, centred=False)) # (1,h,w)
    # np.save('Mask.npy',Mask)
    # Mask = np.load('Mask.npy')
    imgdata = itk.ReadImage('mask.nii')
    Mask = itk.GetArrayFromImage(imgdata)
    Mask = torch.from_numpy(Mask).to(args.device)
    # all_k = torch.zeros(x.shape,dtype=torch.complex64).to(args.device)
    # us_img = torch.zeros(x.shape,dtype=torch.complex64).to(args.device)
    # 固定欠采样矩阵！
    for i in range(shape[0]):
        mask[i,:,:] = Mask
    #     all_k[i,:,:] = myFFT2.image_fft2_k_torch(x[i,:,:])
    #     under_k = (all_k *mask)
    #     us_img[i,:,:] = myFFT2.k_ifft2_image_torch(under_k[i,:,:])

    all_k = torch.fft.ifftshift(torch.fft.fft2(x)).to(args.device)

    under_k = (all_k *mask)

    us_img = torch.fft.ifft2(torch.fft.fftshift(under_k))

    # plt.subplot(121)
    # plt.imshow(torch.abs(mask[0,:,:]).cpu().detach().numpy())
    # plt.subplot(122)
    # plt.imshow(torch.abs(us_img[0,:,:]).cpu().detach().numpy(),'gray')
    # plt.show()
    
    return us_img,all_k,under_k,mask


def create_data( args):
    dev_data = Nii_Data(
        root=r'test',
        transform= DataTransform(),
        batch_size=args.batch_size
    )
    return dev_data

def create_data_loader(args):
    test_data = create_data(args)
    dev_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return dev_loader

def Show(model):
        
        imgdata = itk.ReadImage(r'test\test_gt.nii')
        data = itk.GetArrayFromImage(imgdata)
        data = torch.from_numpy(data.copy())
        gt_img = data[11].unsqueeze(0).to(args.device)
        # imgdata = itk.ReadImage('lung_mask.nii')
        # lung_mask = itk.GetArrayFromImage(imgdata)[11]
        for i in range(gt_img.shape[0]):
            gt_img[i,...]=gt_img[i,...]/torch.max(torch.abs(gt_img[i,...]))

        us_img,all_k,us_k,mask = prep_input(gt_img)

        mask = torch.cat([mask.unsqueeze(1),mask.unsqueeze(1)],dim=1)

        out = model(us_img)
        out = myFFT2.real2complex(out)

        out = out.detach().cpu().numpy()
        gt_img = gt_img.cpu().numpy()
        us_img = us_img.cpu().numpy()
        mask = mask.cpu().numpy()

        # # 保存
        # scio.savemat('gt.mat', {'gt':gt_img[0,:,:]}) 
        # scio.savemat('zf.mat', {'zf':us_img[0,:,:]}) 
        # scio.savemat('Casnet.mat', {'out':out[0,:,:]}) 



        plt.subplot(121)
        # plt.title("recon_img")
        plt.axis('off')
        plt.imshow(np.abs(out[0,:,:]),'gray')
        # plt.subplot(122)
        # # plt.title("recon_img")
        # plt.axis('off')
        # plt.imshow(np.abs(lung_mask),'gray')
        plt.show()

        norm_img_recon = metric.normalized(np.abs(out))
        norm_gt_img = metric.normalized(np.abs(gt_img))
        norm_us_img = metric.normalized(np.abs(us_img))

        psnr_recon = metric.psnr(norm_img_recon[0], norm_gt_img[0])
        psnr_zf = metric.psnr(norm_us_img[0], norm_gt_img[0])

        ssim_recon = ssim(norm_img_recon[0], norm_gt_img[0], data_range=norm_img_recon.max() - norm_img_recon.min())
        ssim_zf = ssim(norm_us_img[0], norm_gt_img[0], data_range=norm_us_img.max() - norm_us_img.min())

        print('recon_psnr=', psnr_recon, 'recon_ssim', ssim_recon, 'zf_psnr', psnr_zf, 'zf_ssim', ssim_zf)

def test_recon(args, model, data_loader):
    st = time.perf_counter()
    losses = []
    num = 0
    PSNR_zf = []
    SSIM_zf = []
    PSNR_recon = []
    SSIM_recon = []
    for iter, data in enumerate(data_loader):
        gt_img = data[0].to(args.device)
        lung_mask = data[1].numpy()

        for i in range(gt_img.shape[0]):
            gt_img[i,...]=gt_img[i,...]/torch.max(torch.abs(gt_img[i,...]))

        us_img,all_k,us_k,mask = prep_input(gt_img)


        mask = torch.cat([mask.unsqueeze(1),mask.unsqueeze(1)],dim=1)

        out = model(us_img)
        out = myFFT2.real2complex(out)

        out = out.detach().cpu().numpy()
        gt_img = gt_img.cpu().numpy()
        us_img = us_img.cpu().numpy()
        mask = mask.cpu().numpy()

        Show(model)


        # plt.subplot(121)
        # plt.imshow(np.abs(lung_mask[1,:,:]),'gray')
        # plt.subplot(122)
        # plt.imshow(np.abs(gt_img[1,:,:]),'gray')
        # plt.show()

        # plt.subplot(141)
        # plt.title("us_img")
        # plt.axis('off')
        # plt.imshow(np.abs(us_img[0,:,:]),'gray')    # 全采样图像
        # plt.subplot(142)
        # plt.title("gt_img")
        # plt.axis('off') 
        # plt.imshow(np.abs(gt_img[0,:,:]),'gray')
        # plt.subplot(143)
        # plt.title("recon_img")
        # plt.axis('off')
        # plt.imshow(np.abs(out[0,:,:]),'gray')
        # plt.subplot(144)
        # plt.title("mask")
        # plt.axis('off')
        # plt.imshow(np.abs(mask[0,0,:,:]),'gray')
        # plt.show()



        #计算指标
        norm_img_recon = metric.normalized(np.abs(out))
        norm_gt_img = metric.normalized(np.abs(gt_img))
        norm_us_img = metric.normalized(np.abs(us_img))

        # ssdu_mask = ssdu_masks()
        # mask1, mask2 = ssdu_mask.Gaussian_selection(kspace.cpu().numpy(), mask_omega.detach().cpu().numpy())
        # savepath_lambda_mask_gaussian = r'D:\MyProject\MystyleGAN\result\mask\gaussian\lambda\{}.png'.format(num)
        # savepath_theta_mask_gaussian = r'D:\MyProject\MystyleGAN\result\mask\gaussian\theta\{}.png'.format(num)
        # plt.imsave(savepath_lambda_mask_gaussian, mask1, cmap='gray')
        # plt.imsave(savepath_theta_mask_gaussian, mask2, cmap='gray')
        #
        # mask1_r, mask2_r = ssdu_mask.uniform_selection(kspace.cpu().numpy(), mask_omega.detach().cpu().numpy())
        # savepath_lambda_mask_random = r'D:\MyProject\MystyleGAN\result\mask\random\lambda\{}.png'.format(num)
        # savepath_theta_mask_random = r'D:\MyProject\MystyleGAN\result\mask\random\theta\{}.png'.format(num)
        # plt.imsave(savepath_lambda_mask_random, mask1_r, cmap='gray')
        # plt.imsave(savepath_theta_mask_random, mask2_r, cmap='gray')
        #
        # savepath_lambda_mask_Net = r'D:\MyProject\MystyleGAN\result\mask\net\lambda\{}.png'.format(num)
        # savepath_theta_mask_Net = r'D:\MyProject\MystyleGAN\result\mask\net\theta\{}.png'.format(num)
        # savepath_mask = r'D:\MyProject\MystyleGAN\result\mask\mask\{}.png'.format(num)
        # plt.imsave(savepath_lambda_mask_Net, mask_lambda.detach().cpu().numpy(), cmap='gray')
        # plt.imsave(savepath_theta_mask_Net, mask_omega.detach().cpu().numpy()-mask_lambda.detach().cpu().numpy(), cmap='gray')
        # plt.imsave(savepath_mask, mask_omega.detach().cpu().numpy(), cmap='gray')

        shape = out.shape
        for i in range(shape[0]):
            psnr_recon = metric.psnr(norm_img_recon[i], norm_gt_img[i])
            psnr_zf = metric.psnr(norm_us_img[i], norm_gt_img[i])
            PSNR_recon.append(psnr_recon)
            PSNR_zf.append(psnr_zf)

            ssim_recon = ssim(norm_img_recon[i], norm_gt_img[i], data_range=norm_img_recon.max() - norm_img_recon.min())
            ssim_zf = ssim(norm_us_img[i], norm_gt_img[i], data_range=norm_us_img.max() - norm_us_img.min())
            SSIM_recon.append(ssim_recon)
            SSIM_zf.append(ssim_zf)

        num = num + 1

    return np.mean(PSNR_recon), np.mean(SSIM_recon), np.mean(PSNR_zf), np.mean(SSIM_zf)


def build_model(args):
    model = UNetModel(2, 2, 64, 4, 0).to(args.device)
    return model



def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    # weight = checkpoint['weight']
    weight = 0
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    return model, weight


def main(args):
    model, weight = load_model(args.checkpoint)
    data_loader = create_data_loader(args)
    num = 0
    mean_PSNR_recon = []
    mean_SSIM_recon = []
    mean_PSNR_zf = []
    mean_SSIM_zf = []
    PSNR_recon, SSIM_recon, PSRN_zf, SSIM_zf = test_recon(args, model, data_loader)
    num = num + args.batch_size
    mean_PSNR_recon.append(PSNR_recon)
    mean_SSIM_recon.append(SSIM_recon)
    mean_PSNR_zf.append(PSRN_zf)
    mean_SSIM_zf.append(SSIM_zf)
    print('mean recon_psnr=', np.mean(mean_PSNR_recon), 'recon_ssim', np.mean(mean_SSIM_recon), 'zf_psnr', np.mean(mean_PSNR_zf), 'zf_ssim', np.mean(mean_SSIM_zf))



def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int, help='Mini batch size')
    parser.add_argument('--sparsity1', type=float, default=0.25, help='The first mask')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Which device to train on. Set to "cuda:n" ,n represent the GPU number')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=False,
                        default=r"checkpoints\model.pt", help='required=True时, 输入模型路径')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
