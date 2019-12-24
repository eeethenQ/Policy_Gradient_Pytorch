import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import datetime


transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((30, 30)),
            torchvision.transforms.ToTensor()
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_screenshot(observation):
    print("Saving to ./final_screenshot.png")
    if observation.shape[0] == 3 or observation.shape[0] == 4:
        observation = observation.transpose((1,2,0))
    plt.imsave("final_screenshot.png", observation)


def get_observation(obs):
    obs = obs.transpose((2,0,1))
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = torch.from_numpy(obs)

    return transform(obs).unsqueeze(0).to(device)

def get_observation_for_pixel_cartpole(env):
    # From 3x400x600 to 3x160x360
    # Same as the Pytorch DQN tutorial, to verify the feasibility of using pixel value to train Cartpole
    def get_cart_location(screen_width):
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.)
    
    obs = env.render(mode='rgb_array')
    obs = obs.transpose((2,0,1))
    _, height, width = obs.shape
    
    # Cart is in the lower half, so strip off the top and bottom of the screen
    obs = obs[:, int(height * 0.4):int(height * 0.8)]

    # Cut the width into three parts, find the one the cart in
    view_width = int(width * 0.6)
    cart_location = get_cart_location(width)
    if cart_location < view_width // 2: # left
        slice_range = slice(view_width)
    elif cart_location > (width - view_width//2): # right
        slice_range = slice(-view_width, None)
    else: # middle
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    # Strip off the edges
    obs = obs[:, :, slice_range]

    # other misc processing
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = torch.from_numpy(obs)
    return transform(obs).unsqueeze(0).to(device)

    
def get_date():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))