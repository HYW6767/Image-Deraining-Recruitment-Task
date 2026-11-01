import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class DerainDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=128):
        self.root_dir = root_dir
        self.is_train = is_train

        data_folder = 'train' if is_train else 'test'
        self.rainy_dir = os.path.join(self.root_dir, data_folder, 'input')
        self.clean_dir = os.path.join(self.root_dir, data_folder, 'target')

        self.rainy_images = sorted(os.listdir(self.rainy_dir))

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                # transforms.RandomCrop(image_size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.rainy_images[idx])

        rainy_image = Image.open(rainy_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')

        rainy_tensor = self.transform(rainy_image)
        clean_tensor = self.transform(clean_image)

        return rainy_tensor, clean_tensor, self.rainy_images[idx]

class DerainDataset_noresize(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=128):
        self.root_dir = root_dir
        self.is_train = is_train

        data_folder = 'train' if is_train else 'test'
        self.rainy_dir = os.path.join(self.root_dir, data_folder, 'input')
        self.clean_dir = os.path.join(self.root_dir, data_folder, 'target')

        self.rainy_images = sorted(os.listdir(self.rainy_dir))

        if self.is_train:
            self.transform = transforms.Compose([
                # transforms.Resize((image_size, image_size)),
                # transforms.RandomCrop(image_size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((196,196)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.rainy_images[idx])

        rainy_image = Image.open(rainy_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')

        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                rainy_image, (196,196))
            rainy_image = TF.crop(rainy_image, i, j, h, w)
            clean_image = TF.crop(clean_image, i, j, h, w)

        rainy_tensor = self.transform(rainy_image)
        clean_tensor = self.transform(clean_image)

        return rainy_tensor, clean_tensor, self.rainy_images[idx]