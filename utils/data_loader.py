from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models.ProtoNet as ProtoNet
from torch.utils.data import Dataset
from PIL import Image
import os

def check_config_parameter(config, parameter):
    keys = parameter.split('.')
    current = config
    for key in keys:
        if key not in current:
            return False
        current = config[key]
    return True

class StandardDataset(Dataset):
    def __init__(self, root, split=None, transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.data = []
        self.targets = []
        self.classes = []

        # Load all image paths and labels
        for class_name in os.listdir(self.root):
            class_path = os.path.join(self.root, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                class_index = len(self.classes) - 1
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.data.append(img_path)
                        self.targets.append(class_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) > 0:
            support_data_idx, query_data_idx = idx
            support_img_path = self.data[support_data_idx]
            query_img_path = self.data[query_data_idx]
            try:
                support_image = Image.open(support_img_path)
                query_image = Image.open(query_img_path)
            except IOError:
                raise ValueError(f"Error opening image file: {support_img_path} or {query_img_path}")
            support_label = self.targets[support_data_idx]
            query_label = self.targets[query_data_idx]

            if self.transform:
                support_image = self.transform(support_image)
                query_image = self.transform(query_image)

            return (support_image, query_image, support_label), query_label
        
        else:
            img_path = self.data[idx]
            try:
                image = Image.open(img_path)
            except IOError:
                raise ValueError(f"Error opening image file: {img_path}")
            label = self.targets[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

def get_dataloaders(config):
    
    dataset_name = config['experiment']['dataset']
    dataset_dir = config['paths']['dataset_dir']
    model_name = config['model']['name']
    resize_size = config['model'].get('resize', None)
    batch_size = config['training'].get('batch_size', None)
    num_workers = config['training'].get('num_workers', 4)
    input_channels = config['model']['input_channels']
    mean = config['model'].get('mean', input_channels * (0.5,))
    std = config['model'].get('std', input_channels * (0.5,))

    transform_list = []
    if resize_size:
        resize_size = config['model']['resize']
        transform_list.append(transforms.Resize((resize_size, resize_size)))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform = transforms.Compose(transform_list)
    
    if dataset_name in ['miniImageNet', 'Fewshot-CIFAR100', 'SAR_class_3', 'SAR_class_6']:
        train_dataset = StandardDataset(root=f'{dataset_dir}/{dataset_name}', split='train', transform=transform)
        test_dataset = StandardDataset(root=f'{dataset_dir}/{dataset_name}', split='test', transform=transform)
        # if val folder does not exist, use test folder as val folder
        if os.path.exists(f'{dataset_dir}/{dataset_name}/val'):
            val_dataset = StandardDataset(root=f'{dataset_dir}/{dataset_name}', split='val', transform=transform)
        else:
            print(f'Since val folder does not exist, using test data as val data')
            val_dataset = StandardDataset(root=f'{dataset_dir}/{dataset_name}', split='test', transform=transform)

        # Check if the number of classes is greater than n_ways
        if check_config_parameter(config.get('experiment', {}), 'train_n_ways') and len(train_dataset.classes) < config['experiment']['train_n_ways']:
            raise ValueError(f"Number of classes in train split is {len(train_dataset.classes)}")
        if check_config_parameter(config.get('experiment', {}), 'val_n_ways') and len(val_dataset.classes) < config['experiment']['val_n_ways']:
            raise ValueError(f"Number of classes in val split is {len(val_dataset.classes)}")
        if check_config_parameter(config.get('experiment', {}), 'test_n_ways') and len(test_dataset.classes) < config['experiment']['test_n_ways']:
            raise ValueError(f"Number of classes in test split is {len(test_dataset.classes)}")
    else:
        raise ValueError(f'Unknow dataset:{dataset_name}')
    

    if model_name == 'ProtoNet':
        train_sampler = ProtoNet.PrototypicalBatchSampler(config, train_dataset.targets, mode='train')
        val_sampler = ProtoNet.PrototypicalBatchSampler(config, val_dataset.targets, mode='val')
        test_sampler = ProtoNet.PrototypicalBatchSampler(config, test_dataset.targets, mode='test')

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader