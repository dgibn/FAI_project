import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io
import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset,DataLoader, random_split
from PIL import Image
import torch.nn.functional as F 

class PACS(Dataset):
    def __init__(self, source_domain, transform):
        self.loader = default_loader
        self.transform = transform
        self.data = []
        self.labels = []

        self.path = f"/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/PACS/{source_domain}"

        # List all class names (subdirectories)
        self.classes = sorted(os.listdir(self.path))  # Ensure consistent class order
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect image paths and corresponding labels
        for cls_name in self.classes:
            cls_dir = os.path.join(self.path, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.data.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_idx = self.labels[idx]

        image = self.transform(Image.open(image_path).convert('RGB'))

        # One-hot encode the label
        label = torch.tensor(label_idx)

        return image, label
       

# def subsample_dataset(dataset, idxs):

#     mask = np.zeros(len(dataset)).astype('bool')
#     mask[idxs] = True

#     dataset.samples = [(p, t) for i, (p, t) in enumerate(zip(dataset.images,dataset.target)) if i in idxs]
#     dataset.uq_idxs = dataset.uq_idxs[mask]

#     return dataset

# def subsample_classes(dataset, include_classes=range(45)):

#     cls_idxs = [i for i, (p, t) in enumerate(zip(dataset.images,dataset.target)) if t in include_classes]

#     # TODO: Don't transform targets for now
#     # target_xform_dict = {}
#     # for i, k in enumerate(include_classes):
#     #     target_xform_dict[k] = i

#     dataset = subsample_dataset(dataset, cls_idxs)

#     # dataset.target_transform = lambda x: target_xform_dict[x]

#     return dataset

# def get_train_val_indices(train_dataset, val_split=0.2):

#     all_targets = [t for i, (p, t) in enumerate(zip(train_dataset.images,train_dataset.target))]
#     train_classes = np.unique(all_targets)

#     # Get train/test indices
#     train_idxs = []
#     val_idxs = []
#     for cls in train_classes:
#         cls_idxs = np.where(all_targets == cls)[0]

#         v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
#         t_ = [x for x in cls_idxs if x not in v_]

#         train_idxs.extend(t_)
#         val_idxs.extend(v_)

#     return train_idxs, val_idxs

def build_dataloader(train_transform,split):
    np.random.seed(42)
    domain = "photo"
    
    total_dataset = PACS(source_domain=domain, transform=train_transform)
    
    total_size = len(total_dataset)
    train_size = int(split * total_size)
    val_size = total_size - train_size
    test_size = int(val_size*0.7)
    
    # Assuming availability of a method like random_split or manual splitting
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

    val_dataset, test_dataset = random_split(val_dataset, [val_size - test_size, test_size])

    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    return train_loader,  val_loader, test_loader
