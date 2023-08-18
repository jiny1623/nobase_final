import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, augmentor = None):
        self.isforward = isforward
        self.max_key = max_key
        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False
            
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input) # 원래는 to_tensor(input * mask)
        
        # stack before augmentation
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        # Apply augmentations if needed
        
#         print("BEFORE: ", kspace.shape, target.shape)
        if self.use_augment:
#             print("HELLO")
            if self.augmentor.schedule_p() > 0.0:
#                 print("CHECK")
                kspace, target = self.augmentor(kspace, target.shape)
#         print("AFTER: ", kspace.shape, target.shape)

#         kspace = kspace * to_tensor(mask)
#         print(kspace.shape)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        kspace = kspace * mask
        return mask, kspace, target, maximum, fname, slice
