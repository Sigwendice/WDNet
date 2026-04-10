import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from trainNetworkHelper import SimpleDiffusionTrainer
from diffusion import DiffusionModel
from dataset import ImageDataset
from unet import Unet
from train_hfrm import HFRM

device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 64
channels = 1
batch_size = 32
lambda_step = 0.1

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomCrop(size=(64, 64))]
)

train_folder = "../datasets/ffhq/train"
train_data = ImageDataset(train_folder, transform=transform)
# # 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

sr = 5
model1 = HFRM(sr=sr).to(device)
model1.load_state_dict(torch.load(f"./models/BinA/A_best5.pth", map_location=device))
# 冻结 net1 的所有参数
for param in model1.parameters():
    param.requires_grad = False  # 冻结 net1 的所有参数

epoches = 500

denoise_model = Unet(
    dim=32,
)

timesteps = 1000
schedule_name = "sigmoid_beta_schedule"
DDPM = DiffusionModel(
    schedule_name=schedule_name,
    timesteps=timesteps,
    beta_start=0.0001,
    beta_end=0.02,
    denoise_model=denoise_model,
).to(
    device
)  # cuda:0

optimizer = Adam(DDPM.parameters(), lr=1e-4)

Trainer = SimpleDiffusionTrainer(
    epoches=epoches,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    timesteps=timesteps,
    IFadjust_learning_rate=True,
    types="type1",
    sr=sr,
)
# 如果模型已经训练好则注释下面这行代码，反之则注释上面两行代码
model = Trainer(model1=model1, model2=DDPM, start_epochs=0)
