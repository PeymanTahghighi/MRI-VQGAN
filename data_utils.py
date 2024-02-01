import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def window_center_adjustment(img):

    hist = np.histogram(img.ravel(), bins = int(np.max(img)))[0];
    hist = hist / (hist.sum()+1e-4);
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/(max_intensity + 1e-4));
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;
def retarget_image(mask, img):
    rows = np.sum(mask, 1);
    rows_thresh = np.where(rows>0);
    first_row = rows_thresh[0][0];
    last_row = rows_thresh[0][-1];

    columns = np.sum(mask, 0);
    columns_thresh = np.where(columns > 0);
    first_column = columns_thresh[0][0];
    last_column = columns_thresh[0][-1];

    return img[first_row:last_row, first_column:last_column];
def cache_dataset(root):
    counter = 0;
    if os.path.exists('cache') is False:
        os.mkdir('cache');
    folders = glob(f'{root}/*/')
    for f in folders:
        first_mri = nib.load(os.path.join(f, 'flair_time01_on_middle_space.nii.gz'));
        second_mri = nib.load(os.path.join(f, 'flair_time02_on_middle_space.nii.gz'));
        brain_mask = nib.load(os.path.join(f, 'brain_mask.nii.gz'));
        first_mri_data = first_mri.get_fdata();
        second_mri_data = second_mri.get_fdata();
        brain_mask_data = brain_mask.get_fdata();
        first_mri_data = window_center_adjustment(first_mri_data);
        second_mri_data = window_center_adjustment(second_mri_data);
        step = 4;
        for i in range(0, brain_mask_data.shape[0], step):
            if np.count_nonzero(brain_mask_data[i,:,:]) > 10000:
                plt.imsave(os.path.join('cache', f'{counter}.png'),retarget_image(brain_mask_data[i,:,:], first_mri_data[i,:,:]),cmap='gray');
                counter += 1;
                plt.close('all');
                # plt.imsave(os.path.join('cache', f'{counter}.png'),retarget_image(brain_mask_data[i,:,:], second_mri_data[i,:,:]),cmap='gray');
                # counter += 1;
                # plt.close('all');


        for j in range(0, brain_mask_data.shape[1], step):
            if np.count_nonzero(brain_mask_data[:,j,:]) > 10000:
                plt.imsave(os.path.join('cache', f'{counter}.png'), retarget_image(brain_mask_data[:,j,:], first_mri_data[:,j,:]),cmap='gray');
                counter += 1;
                plt.close('all');
                # plt.imsave(os.path.join('cache', f'{counter}.png'), retarget_image(brain_mask_data[:,j,:], second_mri_data[:,j,:]),cmap='gray');
                # counter += 1;
                # plt.close('all');

        for k in range(0, brain_mask_data.shape[2], step):
            if np.count_nonzero(brain_mask_data[:,:,k]) > 10000:
                plt.imsave(os.path.join('cache', f'{counter}.png'),retarget_image(brain_mask_data[:,:,k], first_mri_data[:,:,k]),cmap='gray');
                counter += 1;
                plt.close('all');
                # plt.imsave(os.path.join('cache', f'{counter}.png'),retarget_image(brain_mask_data[:,:,k], second_mri_data[:,:,k]),cmap='gray');
                # counter += 1;
                # plt.close('all');

class MRIDataset(Dataset):
    def __init__(self, img_list, args) -> None:
        super().__init__();
        self.img_list = img_list;

        self.transform = A.Compose(
            [
                A.Normalize(),
                A.Resize(args.image_size, args.image_size),
                ToTensorV2()
            ]
        )
    def __len__(self):
        return len(self.img_list);

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index], cv2.IMREAD_COLOR);
        img = self.transform(image = img)['image'];
        return img;

def load_data(args):
    total_mri_samples = list(glob(f'{args.dataset_path}/*'));
    total_mri_samples = shuffle(total_mri_samples, random_state = 42);
    train_set, test_set = total_mri_samples[:int(len(total_mri_samples)*0.8)], total_mri_samples[int(len(total_mri_samples)*0.8)+1:]
    train_data = MRIDataset(train_set[:5] if args.baby_dataset else train_set, args);
    test_data = MRIDataset(test_set[:5] if args.baby_dataset else test_set, args);
    train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return train_loader, test_loader
