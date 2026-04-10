import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from networkHelper import extract
from varianceSchedule import VarianceSchedule

from wave_t import DWT


class DiffusionModel(nn.Module):
    def __init__(
        self,
        schedule_name="linear_beta_schedule",
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        denoise_model=None,
    ):
        super(DiffusionModel, self).__init__()

        self.denoise_model = denoise_model
        self.dwt = DWT()
        # 方差生成
        variance_schedule_func = VarianceSchedule(
            schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end
        )
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)
        # print(x_start.device)     #训练集在cuda上，测试集在cpu上
        # print(t.device)           #cuda
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l1", x_self_cond=None):
        l, h = self.dwt(x_start)
        from torchvision.utils import save_image

        if noise is None:
            noise = torch.randn_like(l)

        x_noisy = self.q_sample(x_start=l, t=t, noise=noise)
        # save_image(x_noisy, "./result/test2/x_noisy.png")
        predicted_noise = self.denoise_model(x_noisy, t, x_self_cond)  #

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index, x_start_noise=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        if x_start_noise != None:
            predicted_noise = self.denoise_model(x, t, x_self_cond=x_start_noise)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start_noise=None):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]

        # start from pure noise (for each example in the batch)
        # if x_start_noise == None:

        img = torch.randn(shape, device=device)  # 随机高斯噪声 从这里开始采样

        # else:
        #     t = torch.full((b,), self.timesteps-1, device=device, dtype=torch.long)
        #     img = self.q_sample(x_start=x_start_noise, t=t)

        imgs = []

        if x_start_noise != None:
            x_start_noise = x_start_noise.to(device)
            for i in tqdm(
                reversed(range(0, self.timesteps)),
                desc="sampling loop time step",
                total=self.timesteps,
            ):
                img = self.p_sample(
                    img,
                    torch.full((b,), i, device=device, dtype=torch.long),
                    i,
                    x_start_noise=x_start_noise,
                )  #
                # imgs.append(img.cpu().numpy())
                imgs.append(img)
        return imgs

    @torch.no_grad()
    def sample(
        self, image_size, batch_size=16, channels=1, x_start_noise=None
    ):  # x_start_noise：HFRM输出的三个高频子带
        if x_start_noise != None:
            return self.p_sample_loop(
                shape=(batch_size, channels, image_size, image_size),
                x_start_noise=x_start_noise,
            )

    def forward(self, mode, **kwargs):
        if mode == "train":
            # 先判断必须参数
            if "x_start" and "t" in kwargs.keys():
                # 接下来判断一些非必选参数
                if "loss_type" and "noise" in kwargs.keys():
                    return self.compute_loss(
                        x_start=kwargs["x_start"],
                        t=kwargs["t"],
                        noise=kwargs["noise"],
                        loss_type=kwargs["loss_type"],
                    )
                elif "loss_type" and "x_self_cond" in kwargs.keys():
                    return self.compute_loss(
                        x_start=kwargs["x_start"],
                        t=kwargs["t"],
                        loss_type=kwargs["loss_type"],
                        x_self_cond=kwargs["x_self_cond"],
                    )
                elif "loss_type" in kwargs.keys():
                    return self.compute_loss(
                        x_start=kwargs["x_start"],
                        t=kwargs["t"],
                        loss_type=kwargs["loss_type"],
                    )
                elif "noise" in kwargs.keys():
                    return self.compute_loss(
                        x_start=kwargs["x_start"], t=kwargs["t"], noise=kwargs["noise"]
                    )
                elif "x_self_cond" in kwargs.keys():
                    return self.compute_loss(
                        x_start=kwargs["x_start"],
                        t=kwargs["t"],
                        x_self_cond=kwargs["x_self_cond"],
                    )
                else:
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"])

            else:
                raise ValueError("扩散模型在训练时必须传入参数x_start和t！")

        elif mode == "generate":
            if "image_size" and "batch_size" and "channels" in kwargs.keys():
                # 接下来是非必选参数
                if "x_start_noise" in kwargs.keys():
                    return self.sample(
                        image_size=kwargs["image_size"],
                        batch_size=kwargs["batch_size"],
                        channels=kwargs["channels"],
                        x_start_noise=kwargs["x_start_noise"],
                    )
                else:
                    return self.sample(
                        image_size=kwargs["image_size"],
                        batch_size=kwargs["batch_size"],
                        channels=kwargs["channels"],
                    )
            else:
                raise ValueError(
                    "扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数"
                )
        else:
            raise ValueError("mode参数必须从{train}和{generate}两种模式中选择")
