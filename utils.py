import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt


transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((20, 20)),
            torchvision.transforms.ToTensor()
        ])

def save_screenshot(observation):
    print("Saving to ./final_screenshot.png")
    plt.imsave("final_screenshot.png", observation)


def get_observation(obs):
    obs = obs.transpose((2,0,1))
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = torch.from_numpy(obs)

    return transform(obs).unsqueeze(0)