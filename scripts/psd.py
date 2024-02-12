import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="100, 0, 100, 0, 100, 0, 100, 0",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/psd-samples"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    config = OmegaConf.load(
        "logs/2023-12-18T09-32-24_psd-ldm-vq-4/configs/2023-12-18T09-32-24-project.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "logs/2023-12-18T09-32-24_psd-ldm-vq-4/checkpoints/epoch=000380.ckpt")  # TODO: check path

    #logs/2023-12-15T10-46-32_psd-ldm-vq-4/checkpoints/epoch=000200.ckpt
    #logs/2023-12-15T14-25-31_psd-ldm-vq-4/checkpoints/epoch=000290.ckpt
    #logs/2023-12-18T09-32-24_psd-ldm-vq-4/checkpoints/epoch=000380.ckpt
    #"logs/2023-12-14T11-26-47_psd-ldm-vq-4/checkpoints/epoch=000110.ckpt"
    # "logs/2023-12-12T15-51-13_psd-ldm-vq-4/checkpoints/epoch=000058.ckpt"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt
    prompt = prompt.split(",")
    prompt = torch.tensor([[[float(value) for value in prompt]]] * opt.n_samples)
    prompt = prompt[..., None]
    prompt = rearrange(prompt, 'b h w c -> b c h w')
    prompt = prompt.to(device)

    uc_prompt = "0,0,0,0,0,0,0,0"
    uc_prompt = uc_prompt.split(",")
    uc_prompt = torch.tensor([[[float(value) for value in uc_prompt]]] * opt.n_samples)
    uc_prompt = uc_prompt[..., None]
    uc_prompt = rearrange(uc_prompt, 'b h w c -> b c h w')
    uc_prompt = uc_prompt.to(device)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples = list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(uc_prompt)

            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(prompt
                                                   # opt.n_samples * prompt
                                                   )
                shape = [3, opt.H // 8, opt.W // 8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 # unconditional_guidance_scale=opt.scale,
                                                 # unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                all_samples.append(x_samples_ddim)

    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
