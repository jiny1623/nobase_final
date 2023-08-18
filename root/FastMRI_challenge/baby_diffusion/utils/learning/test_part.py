import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet

import fastmri as fastmri
import config_file
import losses
import sampling
import sde_lib
import controllable_generation
from models import ddpm, ncsnv2, ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector)
import time

def test(args, score_model, pc_inpainter, data_loader):
    reconstructions = defaultdict(dict)
    start_iter = start = time.perf_counter()
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            
            b, c, h, w, two = kspace.shape
            assert two == 2
            kspace = kspace.view(b*c, h, w, 2).permute(0, 3, 1, 2)
            kspace = kspace[:, :, :, w//2 - 192 : w//2 + 192]
            mask = mask.view(1, 1, 1, -1)
            mask = mask[:, :, :, w//2 - 192 : w//2 + 192]
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            output = pc_inpainter(score_model, kspace, mask)
            output = output.permute(0, 2, 3, 1).view(b, c, h, 384, 2)
            result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(output)), dim=1)
            height = result.shape[-2]
            width = result.shape[-1]
            output = result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]
            
#             print(
#                 f'Iter = [{count:4d}/{len(data_loader):4d}] '
#                 f'Time = {time.perf_counter() - start_iter:.4f}s',
#             )
            start_iter = time.perf_counter()
            
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
            break

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())
    config = config_file.get_config()
    
    score_model = mutils.create_model(config)
    
    checkpoint = torch.load(args.exp_dir / 'model20.pt', map_location='cpu') ###########################################################
    print(checkpoint['epoch'])
    score_model.load_state_dict(checkpoint['model'])
    
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
    
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    snr = 0.16
    n_steps = 1
    probability_flow = False
    pc_inpainter = controllable_generation.get_pc_inpainter(sde, predictor, corrector,
                                                            snr=snr,
                                                            n_steps=n_steps,
                                                            probability_flow=probability_flow,
                                                            continuous=config.training.continuous,
                                                            denoise=True)
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, score_model, pc_inpainter, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)