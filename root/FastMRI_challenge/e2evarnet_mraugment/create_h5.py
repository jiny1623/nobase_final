import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import h5py
from pathlib import Path

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions
from utils.model.varnet import VarNet
        
def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, _, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
            
            if iter % 100 == 0:
                print("Iter " + str(iter), ", time ", str(time.perf_counter() - start))
                start = time.perf_counter()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions

def write_h5(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)
    
    model_path = Path('../result/augment_new/checkpoints')
    checkpoint = torch.load(model_path / 'best_model.pt')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])

    print("reconstructing train set...")
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, isforward=True)
    train_reconstructions = validate(args, model, train_loader)
    
    print("writing train reconstructions into h5 file...")
    for fname, recons in train_reconstructions.items():
        with h5py.File(args.out_path_train / fname, 'w') as wf:
            wf.create_dataset('image_varnet', data=recons)
    
    print("reconstructing val set...")
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, isforward=True)
    val_reconstructions = validate(args, model, val_loader)
    
    print("writing val reconstructions into h5 file...")
    for fname, recons in val_reconstructions.items():
        with h5py.File(args.out_path_val / fname, 'w') as wf:
            wf.create_dataset('image_varnet', data=recons)
    
    print("reconstructing leaderboard set... (for evaluation only)")
    acc4_loader = create_data_loaders(data_path = args.data_path_leaderboard_acc4, args = args, isforward=True)
    acc8_loader = create_data_loaders(data_path = args.data_path_leaderboard_acc8, args = args, isforward=True)
    acc4_reconstructions = validate(args, model, acc4_loader)
    acc8_reconstructions = validate(args, model, acc8_loader)
    
    print("writing leaderboard reconstructions into h5 file...")
    for fname, recons in acc4_reconstructions.items():
        with h5py.File(args.out_path_leaderboard_acc4 / fname, 'w') as wf:
            wf.create_dataset('image_varnet', data=recons)
                
    for fname, recons in acc8_reconstructions.items():
        with h5py.File(args.out_path_leaderboard_acc8 / fname, 'w') as wf:
            wf.create_dataset('image_varnet', data=recons)