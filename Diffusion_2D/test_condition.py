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
from ConditionedModel import LabelConditionedModel
from dataset import dataset_condition_brats_2D_test,dataset_condition_brats_2D_shape_test
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def show_images(x):
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1,2,0).clip(0,1) *255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def show_shape(x):
    x[x!=0] = 1
    print(x[128][128])
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1,2,0).clip(0,1)*255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

if __name__ == "__main__":
    device = torch.device('cuda')
    dataset = dataset_condition_brats_2D_shape_test.Train_Dataset(r"E:\HLX\PythonProject\Diffusion_2D_BraTS\condition_label")
    train_dl = DataLoader(dataset, 1, False, num_workers=1)
    sample = torch.randn(1, 4, 240, 240).to(device)
    model = LabelConditionedModel().to(device)
    ckpt = torch.load(r"E:\HLX\PythonProject\Diffusion_2D_BraTS\7_best.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['net'].items() if k.startswith('module.')})

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )
    flag = 0
    for step, batch in enumerate(train_dl):
        # start = time.time()
        clean_images = batch.to(device)
        sample = torch.randn(1, 4, 240, 240).to(device)
        time_flag = time.time()
        image = show_shape(clean_images[0][1])
        image.save(str(time_flag) + '_shape' + '.png')
        for i,t in enumerate(noise_scheduler.timesteps):
            print(t)
            with torch.no_grad():
                residual = model(sample,t,clean_images)
            sample = noise_scheduler.step(residual, t, sample).prev_sample

        print(sample.shape)
        image = show_images(sample[0][0])
        image.save(str(time_flag) +'_0'+'.png')
        image = show_images(sample[0][1])
        image.save(str(time_flag) +'_1'+ '.png')
        image = show_images(sample[0][2])
        image.save(str(time_flag) + '_2' + '.png')
        image = show_images(sample[0][3])
        image.save(str(time_flag) + '_3' + '.png')

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
