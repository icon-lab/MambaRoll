import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L


class MRIDataset(Dataset):
    def __init__(
        self,
        data_dir,
        contrast,
        us_factor,
        stage,
    ):
        self.data_dir = data_dir
        self.contrast = contrast
        self.stage = stage
        self.us_factor = us_factor
        self.name = os.path.basename(os.path.normpath(data_dir))

        self.data = self._load_data()
        self.image_fs = (self.data['image_fs'])[:,None]
        self.image_us = (self.data['image_us'])[:,None]
        self.us_masks = (self.data['us_masks'])[:,None].astype(np.float32)
        self.subject_ids = self.data['subject_ids']
        self.us_factor = self.data['us_factor']
        self.coilmaps = self.data.get('coilmaps')
        
        if self.coilmaps is None:
            self.coilmaps = np.ones_like(self.image_fs)

        # Squeeze redundant dimensions
        if self.image_us.ndim == 5:
            self.image_us = self.image_us.squeeze()

        # Normalization
        self.image_us = self.image_us / np.abs(self.image_fs).max(axis=(-1,-2), keepdims=True)
        self.image_fs = self.image_fs / np.abs(self.image_fs).max(axis=(-1,-2), keepdims=True)
    
    def _load_data(self):
        data_path = os.path.join(self.data_dir, f'us{self.us_factor}x', self.stage, f'{self.contrast}.npz')
        data = np.load(data_path)

        return data
    
    def _load_mask(self):
        mask_path = os.path.join(self.data_dir, 'mask', self.stage)
        files = [f for f in os.listdir(mask_path) if f.endswith('.npy')]

        # Sort by slice index
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        data = []
        for file in files:
            data.append(np.load(os.path.join(mask_path, file)))
        
        return np.array(data).astype(np.float32)

    def __len__(self):
        return len(self.image_fs)

    def __getitem__(self, i):
        return self.image_fs[i], self.image_us[i], self.us_masks[i], self.coilmaps[i], i


class CTDataset(Dataset):
    def __init__(
        self,
        data_dir,
        us_factor,
        stage,
        contrast=None
    ):
        self.data_dir = data_dir
        self.stage = stage
        self.us_factor = us_factor
        self.name = os.path.basename(os.path.normpath(data_dir))
        self.main_dir = os.path.join(data_dir, stage)
        
        data_fs, data_us = self._load_data()
        self.image_fs = data_fs['image_fs']
        self.image_us = data_us['image_us']
        self.sinogram_us = data_us['sinogram_us']
        self.theta = data_us['projection_angles']
        self.subject_ids = data_us['subject_ids']
        self.us_factor = data_us['us_factor']

        # Normalize
        denom = self.image_fs.max(axis=(-1,-2), keepdims=True)
        self.image_fs = self.image_fs / denom
        self.image_us = self.image_us / denom
        self.sinogram_us = self.sinogram_us / denom

    def _load_data(self):
        fs_data = np.load(os.path.join(self.data_dir, self.stage, f'image_fs.npz'))
        us_data = np.load(os.path.join(self.data_dir, self.stage, f'us{self.us_factor}x.npz'))
        return fs_data, us_data

    @property
    def image_size(self):
        return self.image_fs.shape[-2:]

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, i):

        return self.image_fs[i], self.image_us[i], self.sinogram_us[i], self.theta[i], self.us_factor, i


class DataModule(L.LightningDataModule):
    def __init__(
        self, 
        dataset_dir,
        dataset_class,
        contrast,
        us_factor,
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.contrast = contrast
        self.us_factor = us_factor
        self.num_workers = num_workers

        self.dataset_class = globals()[dataset_class]

    def setup(self, stage: str) -> None:
        contrast = self.contrast
        us_factor = self.us_factor

        if stage == "fit":
            self.train_dataset = self.dataset_class(
                data_dir=self.dataset_dir,
                contrast=contrast,
                us_factor=us_factor,
                stage='train'
            )

            self.val_dataset = self.dataset_class(
                data_dir=self.dataset_dir,
                contrast=contrast,
                us_factor=us_factor,
                stage='val'
            )

        if stage == "validate":
            self.val_dataset = self.dataset_class(
                data_dir=self.dataset_dir,
                contrast=contrast,
                us_factor=us_factor,
                stage='val'
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                data_dir=self.dataset_dir,
                contrast=contrast,
                us_factor=us_factor,
                stage='test'
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False
        )
