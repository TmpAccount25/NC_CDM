import time

import torchvision.utils
from diffusers import DDPMPipeline,DDPMScheduler
import cv2
import torch
import torchvision
from PIL import Image
import model
import numpy as np
import time
def show_images(x):
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1,2,0).clip(0,1) *255
    print(grid_im.shape)
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

import SimpleITK as sitk

def show_volumns(x):
    print(x.shape)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu()
    print(grid_im.shape)
    # grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    image = sitk.GetImageFromArray(grid_im)
    return image

def show_volumns_multi(x):
    print(x.shape)
    x_1 = x[0]
    grid_1 = torchvision.utils.make_grid(x_1)
    grid_im_1 = grid_1.detach().cpu()
    x_2 = x[1]
    grid_2 = torchvision.utils.make_grid(x_2)
    grid_im_2 = grid_2.detach().cpu()

    grid_im = np.zeros((2, 16, 96, 96))
    grid_im[0] = grid_im_1
    grid_im[1] = grid_im_2
    grid_im = grid_im.transpose((1,2,3,0))

    # grid_im_complex = np.zeros((16, 96, 96)).astype(np.complex)
    grid_im_complex = grid_im_1 + grid_im_2 * 1j
    grid_im_complex = np.abs(grid_im_complex)
    print(grid_im_complex.shape)
    # grid_im_complex = grid_im_complex.transpose((1, 2, 0))
    print(grid_im_complex.shape)
    # grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))

    image = sitk.GetImageFromArray(grid_im)
    image_complex = sitk.GetImageFromArray(grid_im_complex)
    return image, image_complex

if __name__ == "__main__":
    device = torch.device('cuda')
    sample = torch.randn(1, 2, 16, 96, 96).to(device)
    model = model.model().to(device)
    ckpt = torch.load(r"6_best.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['net'].items() if k.startswith('module.')})

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )
    flag = 0
    for k in range(10000):
        # start = time.time()
        sample = torch.randn(1, 2, 16, 96, 96).to(device)
        for i,t in enumerate(noise_scheduler.timesteps):
            print(t)
            with torch.no_grad():
                residual = model(sample,t).sample
            sample = noise_scheduler.step(residual, t, sample).prev_sample
            # break
        time_flag = time.time()
        print(sample.shape)
        # image = show_images(sample[0][0])
        # image.save(str(time_flag) + '_0' + '.png')
        # image = show_images(sample[0][1])
        # image.save(str(time_flag) + '_1' + '.png')
        image,image_complex = show_volumns_multi(sample[0])
        sitk.WriteImage(image,str(time_flag)  +"_complex"+ '.nii.gz')
        sitk.WriteImage(image_complex, str(time_flag) +"_reconst"+ '.nii.gz')
        break
        # image = show_images(sample[0][2])
        # image.save(str(time_flag) + '_2' + '.png')
        # image = show_images(sample[0][3])
        # image.save(str(time_flag) + '_3' + '.png')

        # end = time.time()
        # print(end-start)
            # print(sample.shape)
            # if flag == 0:
            #     image = show_images(sample)
            #     image.save(str(t) + '.png')
            # if t%100 == 0:
            #     image = show_images(sample)
            #     image.save(str(t)+'.png')
            # flag += 1
