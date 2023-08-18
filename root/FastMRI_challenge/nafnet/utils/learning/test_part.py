import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
# from utils.model.unet import Unet
from utils.model.nafnet import NAFNet
from torchvision.models import *
from torchsummary import summary

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)
    
    with torch.no_grad():
        for (input, grappa, _, _, fnames, slices) in data_loader:
            input = input.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True) 
            print(grappa.shape)
            print(input.shape)
            stacked_input = torch.stack((input, grappa), dim=1)
            print("INPUT:", stacked_input.shape)
            output = model(stacked_input)
            print("OUTPUT:", output.shape)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
    return reconstructions, inputs


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

#     img_channel = 1
#     width = 16

#     enc_blks = [1, 1, 1, 1]
#     middle_blk_num = 1
#     dec_blks = [1, 1, 1, 1]
    
    img_channel = 2
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    
    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
#     summary(model, (2, 384, 384), batch_size=1)
    
#     model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)