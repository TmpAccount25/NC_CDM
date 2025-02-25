import argparse
import shutil
import time
import logging
import numpy as np
import torch
import torchvision
import pathlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from myutils import compressed_sensing as cs

from myutils.my_math import *
from Setdata import Nii_Data
from tqdm import tqdm
import scipy.io as io
import myFFT2
from kikiNet import KIKInet
import SimpleITK as itk

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
    # plt.imshow(torch.abs(under_k[0,:,:]).cpu().detach().numpy())
    # plt.subplot(122)
    # plt.imshow(torch.abs(us_img[0,:,:]).cpu().detach().numpy(),'gray')
    # plt.show()
    
    return us_img,all_k,under_k,mask

###############################################################################
# Data - fast-MRI
###############################################################################

# def create_data_loaders(args):
#     train_data_root = "D:/SCI/knee_self_supervised/singlecoil_train/kspace.h5"

#     train_data=MyDataset(train_data_root)
#     dev_data=MyDataset(train_data_root)

#     train_loader = DataLoader(
#         dataset=train_data,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True
#     )
#     dev_loader = DataLoader(
#         dataset=dev_data,
#         batch_size=args.batch_size,
#         num_workers=0,
#         pin_memory=True
#     )
    
#     print('数据加载完毕！')
#     return train_loader, dev_loader

###############################################################################
# Data - Xe129
###############################################################################

def create_data( args):
    train_data = Nii_Data(
        root=r'.\train',
        transform= DataTransform(),
        batch_size=args.batch_size
    )
    dev_data = Nii_Data(
        root=r'.\test',
        transform= DataTransform(),
        batch_size=args.batch_size
    )
    return dev_data, train_data

def create_data_loaders(args):
    dev_data, train_data = create_data(args)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, dev_loader

def train_epoch(args, epoch, model, epoch_iterator, optimizer, writer, criterion):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(epoch_iterator)

    Loss = []

    for iter, data in enumerate(epoch_iterator):


        gt_img = data.to(args.device)

        # for i in range(gt_img.shape[0]):
        #     gt_img[i,...]=gt_img[i,...]/torch.max(torch.abs(gt_img[i,...]))

        # plt.subplot(133)
        # plt.imshow(torch.abs(gt_img[0,:,:]).cpu().detach().numpy(),'gray')
        # plt.show()

        us_img,all_k,us_k,mask = prep_input(gt_img)


        mask = torch.cat([mask.unsqueeze(1),mask.unsqueeze(1)],dim=1)

        
        out, out1, k_net_out, k_net_out_1 = model(us_img, mask)

        # gt_img = myFFT2.complex2real(gt_img)

        loss_kspace =torch.mean(torch.square( torch.abs(k_net_out - all_k)))
        loss_kspace_1 = torch.mean(torch.square( torch.abs(k_net_out_1 - all_k)))
        loss_mag = torch.mean(torch.square(torch.abs(out) - gt_img)) 
        loss_mag1 = torch.mean(torch.square(torch.abs(out1) - gt_img)) 
        k_loss = loss_kspace + loss_kspace_1
        img_loss = loss_mag + loss_mag1
        loss = k_loss + img_loss
        # loss = criterion(output, gt_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss.append(loss.item())
        writer.add_scalar('Train Loss', loss, global_step + iter)
    return np.mean(Loss)


def evaluate(args, epoch, model, epoch_iterator, writer, criterion):
    model.eval()
    avg_loss = 0.
    avg_PSNR = 0.
    idx = 0
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(epoch_iterator)

    Loss = []
    with torch.no_grad():
        for iter, data in enumerate(epoch_iterator):


            gt_img = data.to(args.device)
            
            # for i in range(gt_img.shape[0]):
            #     gt_img[i,...]=gt_img[i,...]/torch.max(torch.abs(gt_img[i,...]))

            us_img,all_k,us_k,mask = prep_input(gt_img)

            mask = torch.cat([mask.unsqueeze(1),mask.unsqueeze(1)],dim=1)

            out, out1, k_net_out, k_net_out_1 = model(us_img, mask)

            # gt_img = myFFT2.complex2real(gt_img)

            loss_kspace =torch.mean(torch.square( torch.abs(k_net_out - all_k)))
            loss_kspace_1 = torch.mean(torch.square( torch.abs(k_net_out_1 - all_k)))
            loss_mag = torch.mean(torch.square(torch.abs(out) - gt_img)) 
            loss_mag1 = torch.mean(torch.square(torch.abs(out1) - gt_img)) 
            k_loss = loss_kspace + loss_kspace_1
            img_loss = loss_mag + loss_mag1
            loss = k_loss + img_loss

            Loss.append(loss.item())

    ############################ Show ####################################

    plot_epoch=[0,5,207]
    if epoch in plot_epoch:
        plt.subplot(131)
        plt.imshow(torch.abs(us_img[0,:,:]).cpu().detach().numpy(),'gray')    # 全采样图像
        plt.subplot(132)
        plt.imshow(torch.abs(gt_img[0,:,:]).cpu().detach().numpy(),'gray')
        plt.subplot(133)
        plt.imshow(torch.abs(out[0,:,:]).cpu().detach().numpy(),'gray')
        plt.show()
        # data_range=np.amax(torch.abs(gt_img[0,0,:,:]).cpu().detach().numpy()) - np.amin(torch.abs(us_img[7,0,:,:]).cpu().detach().numpy())
        # PSNR = compare_psnr(torch.abs(gt_img[0,0,:,:]).cpu().detach().numpy(),torch.abs(output[7,0,:,:]).cpu().detach().numpy(),data_range=data_range)
        # PSNR_gt = compare_psnr(torch.abs(gt_img[0,0,:,:]).cpu().detach().numpy(),torch.abs(us_img[7,0,:,:]).cpu().detach().numpy(),data_range=data_range)
        # print('PSNR_gt={},PSNR={}'.format(PSNR_gt,PSNR))
        # 单张图片best_model----test_show
        # Show(args.best_model)
    # Test_avgPsnr.Compute(args.best_model,data_loader,args)
    return np.mean(Loss)

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch':epoch,
            'args':args,
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'best_dev_loss':best_dev_loss,
            'exp_dir':exp_dir
        },
        # f = exp_dir/'model.pt'
        f = args.checkpoint
    )
    if is_new_best:
        # shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        shutil.copyfile(args.checkpoint, args.best_model)

def build_model(args):

    # 初始化net
    model = KIKInet(2, 64, 64, 4, 0).to(args.device)
    return model

def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()
    return optimizer, criterion

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    optimizer, criterion = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer, criterion

def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        # 调用哪个model ##########################################################################
        checkpoint, model, optimizer, criterion = load_model(args.checkpoint)

        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        optimizer, criterion = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader = create_data_loaders(args)

    for epoch in range(start_epoch, args.epochs):
        train_iterator = tqdm(train_loader, desc='Train')
        train_loss = train_epoch(args, epoch, model, train_iterator, optimizer, writer, criterion)
        print('Train,epoch=', epoch, 'train_loss=', train_loss)
        dev_iterator = tqdm(dev_loader, desc='Dev')
        dev_loss = evaluate(args, epoch, model, dev_iterator, writer, criterion)
        print('Eval loss', 'train_loss=', dev_loss)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
    writer.close()

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--report_interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this'
                        )
    parser.add_argument('--exp_dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and result should be saved'
                        )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=False,
                        default=r"Xe\model.pt", help='resume=True时, 输入模型路径')
    parser.add_argument('--best_model', type=pathlib.Path, required=False,
                        default=r"Xe\best_model.pt", help='resume=True时, 输入best_model的路径')
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)