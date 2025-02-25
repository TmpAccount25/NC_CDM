import time

from diffusers import UNet2DModel,UNet2DConditionModel,UNet3DConditionModel
import torch
from diffusers import DDPMPipeline,DDPMScheduler
from torch import nn

def model():
    model = UNet2DModel(
            sample_size = 240,
            in_channels = 2,
            out_channels = 2,
            layers_per_block = 2,
            block_out_channels = (64,128,128,256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            )
        )
    return model

class model_Condition(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size = 240,
            in_channels = 6,
            out_channels = 4,
            layers_per_block = 2,
            block_out_channels = (64,128,128,256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            )
        )


    def forward(self, x, t, class_label):
        net_input = torch.cat((x, class_label), 1)
        return self.model(net_input, t).sample

if __name__ == "__main__":
    device = torch.device("cuda")
    start = time.time()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )
    sample = torch.randn(1, 2, 96, 96).to(device)
    clean_images = torch.ones([1, 2, 240,240]).to(device)
    model = model().to(device)
    for i, t in enumerate(noise_scheduler.timesteps):
        print(t)
        with torch.no_grad():
            residual = model(sample, t.to(device))[0]
        sample = noise_scheduler.step(residual, t, sample).prev_sample
    end = time.time()
    print(end-start)
    # print(sample.shape)
    # print("OK")