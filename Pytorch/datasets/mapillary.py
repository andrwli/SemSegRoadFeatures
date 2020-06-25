import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

num_classes = 66
#root = "/mnt/data/"
#root = 'C:\\Users\\Jarvis\\Projects\\SeniorProject\\mapillary_vistas_dataset_v1.1\\'
root = '/home/ec2-user/SageMaker/preprocessed_data/'

palette = [
	165, 42, 42, 
	0, 192, 0,
	196, 196, 196, 
	190, 153, 153, 
	180, 165, 180, 
	90, 120, 150, 
	102, 102, 156, 
	128, 64, 255, 
	140, 140, 200, 
	170, 170, 170, 
	250, 170, 160,
	96, 96, 96, 
	230, 150, 140, 
	128, 64, 128, 
	110, 110, 110, 
	244, 35, 232, 
	150, 100, 100, 
	70, 70, 70, 
	150, 120, 90, 
	220, 20, 60, 
	255, 0, 0, 
	255, 0, 100, 
	255, 0, 200, 
	200, 128, 128, 
	255, 255, 255, 
	64, 170, 64, 
	230, 160, 50, 
	70, 130, 180, 
	190, 255, 255, 
	152, 251, 152, 
	107, 142, 35, 
	0, 170, 30, 
	255, 255, 128, 
	250, 0, 30,
	100, 140, 180, 
	220, 220, 220, 
	220, 128, 128, 
	222, 40, 40,
	100, 170, 30, 
	40, 40, 40, 
	33, 33, 33, 
	100, 128, 160, 
	142, 0, 0, 
	70, 100, 150, 
	210, 170, 100, 
	153,153, 153, 
	128, 128, 128,
	0, 0, 80, 
	250, 170, 30,
	192, 192, 192,
	220, 220, 0,
	140, 140, 20,
	119, 11, 32,
	150, 0, 255,
	0, 60, 100,
	0, 0, 142,
	0, 0, 90,
	0, 0, 230,
	0, 80, 100,
	128, 64, 64,
	0, 0, 110,
	0, 0, 70,
	0, 0, 192,
	32, 32, 32,
	120, 10, 10,
	0, 0, 0
]

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def make_dataset(segmentation_type, mode):
    if segmentation_type == 'semantic':
        mask_dir_name = 'labels'
    else:
        raise ValueError('Segmentation type {} unsupported'.format(segmentation_type))

    img_path = os.path.join(root, mode, 'images')
    mask_path = os.path.join(root, mode, mask_dir_name)
	
    assert len(os.listdir(img_path)) == len(os.listdir(mask_path))
        
    dataset = []
    names = [img.split('.jpg')[0] for img in os.listdir(img_path)]
    for name in names:
        data_item = (os.path.join(img_path, name + '.jpg'), os.path.join(mask_path, name + '.png'))
        dataset.append(data_item)
    return dataset

class Mapillary(Dataset):
    def __init__(self, segmentation_type, mode, joint_transform=None,
			     sliding_crop=None, transform=None, target_transform=None):
				 
        assert (segmentation_type in set(('panoptic', 'semantic', 'instance')))
        assert (mode in set(('training', 'validation')))
		
        self.imgs = make_dataset(segmentation_type, mode)
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, mask_path = self.imgs[idx]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        mask = Image.fromarray(mask_copy.astype(np.uint8))	

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask
        return img, mask
