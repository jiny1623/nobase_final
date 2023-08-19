import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []
        self.recon_path = Path('../recon_data/')
        self.which_data = root.parent.name

        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]
        

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        with h5py.File(fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            grappa = hf['image_grappa'][dataslice]
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
        
        with h5py.File(self.recon_path / 'recon_varnet' / self.which_data / fname.name, "r") as hf:
            varnet = hf['image_varnet'][dataslice]
        with h5py.File(self.recon_path / 'recon_diffusion' / self.which_data / fname.name, "r") as hf:
            diffusion = hf['image_diffusion'][dataslice]
        
        return self.transform(input, grappa, varnet, diffusion, target, attrs, fname.name, dataslice)

def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
