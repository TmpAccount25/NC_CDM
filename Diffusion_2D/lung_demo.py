import torch.utils.data.dataset
import torchvision
from dataset import dataset_lung_2D
from torchvision import transforms
from diffusers import DDPMScheduler
import model
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time

if __name__ == "__main__":
    device = torch.device('cuda')
    dataset = dataset_lung_2D.Train_Dataset(r"Lung")
    train_dl = DataLoader(dataset, 32, False)
    timesteps = torch.linspace(0, 1000, 1).long().to(device)
    model = model.model().to(device)
    model = torch.nn.DataParallel(model, device_ids=[0])
    # if True:
    #     ckpt = torch.load(r"best.pth")
    #     model.load_state_dict(ckpt["net"])
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    losses = []
    loss_flag = 10e+10
    for epoch in range(100000):
        for step, batch in enumerate(train_dl):
            clean_images = batch.to(device)
            # print(clean_images.shape)
            noise = torch.randn(clean_images.shape).to(device)
            batch_size = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,),device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images,noise,timesteps)
            noisy_pred = model(noisy_images,timesteps,return_dict=False)[0]
            loss = F.mse_loss(noisy_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
        if (epoch +1) % 2 == 0:
            loss_last_epoch = sum(losses[-len(train_dl) :]) / len(train_dl)
            print(f"Epoch:{epoch + 1}, loss:{loss_last_epoch}")

        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        if loss_flag > loss:
            torch.save(state,r"best_zero.pth")
            loss_flag = loss



