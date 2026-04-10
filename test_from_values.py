#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 11:13
# @Author  : Atrio
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/5 12:06
# @Author  : Atrio
import torchvision.transforms as transforms
import scipy.io

from networkHelper import *
from value_model import HFRM

# from train_hfrm import HFRM
from wave_t import DWT, IDWT
from torchvision.utils import save_image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_value_conv(mat_path=r"C:\Users\Atrio\Desktop\A\matrixs\hua_y10.mat"):
    # 读取 .mat 文件
    data = scipy.io.loadmat(mat_path)  # 替换为你的 .mat 文件路径
    array = data["Y_test"]  # 替换为你的数组在 .mat 文件中的键名

    # 确保数组的形状为 (M, 1)，其中 M 是 16 的倍数
    M, _ = array.shape
    assert M % 16 == 0, "M 必须是 16 的倍数"

    # 计算新的维度
    num_blocks = M // 16

    # 重塑数组为 (num_blocks, 16)
    reshaped_array = array.reshape(num_blocks, 16)

    # 将每个 16 元素的块转换为 4x4 矩阵
    blocks = reshaped_array.reshape(num_blocks, 4, 4)

    # 添加额外的维度以形成最终的形状为 (num_blocks, 1, 4, 4)
    final_tensor = blocks.reshape(1, num_blocks, 4, 4)

    # 将 NumPy 数组转换为 PyTorch 张量
    final_tensor = torch.from_numpy(final_tensor) / 10.0

    return final_tensor


def load_value(sr, mat_path=r"C:\Users\Atrio\Desktop\A\matrixs\hua_y10.mat"):
    # 读取 .mat 文件
    data = scipy.io.loadmat(mat_path)  # 替换为你的 .mat 文件路径
    array = data["Y_test"]  # 替换为你的数组在 .mat 文件中的键名

    # 确保数组的形状为 (M, 1)，其中 M 是 16 的倍数
    M, _ = array.shape

    # 将 NumPy 数组转换为 PyTorch 张量
    final_tensor = torch.from_numpy(array) / 100.0

    final_tensor = final_tensor.reshape(1, sr * sr * 16, 1, 1)

    return final_tensor


dwt = DWT()
idwt = IDWT()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

image_size = 32
channels = 1


# sr_list = [2,3,4,5,8]
# mr = ['1_6', '3_5', '6_25', '10', '20']

sr_list = [2, 3, 4, 5, 8]
mr = ["1_6", "3_5", "6_25", "10", "25"]

i = 0
for sr in sr_list:
    values = load_value(sr=sr, mat_path=f"E:\实验代码\matrix\hua\hua_y{mr[i]}.mat")

    model1 = HFRM(sr=sr).to(device)
    model1.load_state_dict(torch.load(f"models/F_BIN/A_best{sr}.pth"))

    model2 = f"./models/F_BIN/B_best{sr}.pth"
    DDPM = torch.load(model2)

    model1.eval()
    DDPM.eval()

    with torch.no_grad():

        values = values.to(device)
        # x_cond, xp_l, xp = model1(values)
        output, xp_l, xp = model1.forward_from_phix(values)
        # 拆分特征
        lh, hl, hh = torch.chunk(output, chunks=3, dim=1)

        # 批量生成（关键修改点）
        samples = DDPM(
            mode="generate",
            image_size=image_size,
            batch_size=1,  # 使用实际patch数量作为batch_size
            channels=channels,
            x_start_noise=xp_l,
        )

        # 批量逆变换
        output_patches = idwt(samples[-1], lh, hl, hh)

        save_image(output_patches, f"./result/system/hua{mr[i]}.png")
    i += 1
