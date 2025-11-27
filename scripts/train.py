import argparse
import torch
from torch import nn
import os, sys
from data.CUB import *
from model import *
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from tqdm import tqdm
import torchvision
from torchvision.transforms import v2

parser = argparse.ArgumentParser(description='Train GCD on different datasets.')
parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name, e.g., "OfficeHome", "PACS", "Domain_Net"')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
parser.add_argument('--task_epochs', type=int, default=30, help='Number of epochs for training (default: 20)')
parser.add_argument('--task_learning_rate', type=float, default=0.1, help='Learning rate (default: 0.1)')
parser.add_argument('--device',type=int,default=0,help='Cuda Id')
parser.add_argument('--split',type=str,default='train',help='train/val/test split')

args = parser.parse_args()

dataset_num_classes_mapping = {
    'OfficeHome': 40,
    'PACS': 7,
    'Domain_Net': 250,
    'CUB': 200,
    'CIFAR10': 10,
    'SCARS': 98
}

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

train_transform,test_transform = make_transform(224),make_transform(224)

# train_transform = ContrastiveLearningViewGenerator(train_transform, n_views=2)

link = "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiY3Z1N212czIybTE4eTk3b2Y5a2tvbzBvIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEwMTMzNTB9fX1dfQ__&Signature=IzAsd8SFJVUXhTRpUH3iYEJ%7EML9nPWLalOKpJvmtYgR0bYRvEP9nJ2wRijYdZbPSIDQmQ1kL4Em%7ERryPsP2T4WuMIh7w-a3kCBKkNwADFW1l3LUFxmxs6sgs1idwrap6qm6qw4aaZ68-WtfIyjYB7z7FgWdgtOQcwJn%7E%7EJhjFaKEq2-f9MmnhNkX-5HrJ2ymBKxqadDThHr44xOzs0RZ0vceJhV%7EaZmK929qXR2CDMHhf7AZdZBw1zMUB-ksmt7HgoW048UK7tkju1Po4q2kgu5JfJTemtaJNkRHhhxv2w1o039hoTa%7EslVFMND7jVrgp9A3HALRVYkazoKOkZeGtw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1828217777900514"
REPO_DIR = '/users/student/pg/pg23/vaibhav.rathore/D_GCD/DG/project/dinov3'
global_model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=link)
global_model.head = nn.Identity()
global_model = global_model.to(device)

# with torch.no_grad():
#     out = global_model(torch.randn(1, 3, 224, 224).to(device))
#     print(out.shape)  # should be [1, 768] or similar



for param in global_model.parameters():
    param.requires_grad = False
for param in global_model.blocks[-1].parameters():
    param.requires_grad = True

model = InputModel(in_dim=768,out_dim = dataset_num_classes_mapping[args.dataset_name]).to(device)
task_optimizer = torch.optim.SGD([ {'params': model.parameters(), 'lr': args.task_learning_rate},],momentum=0.9,weight_decay=5e-5)
# Define the cosine annealing learning rate scheduler for task_optimizer
task_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(task_optimizer, T_max=args.task_epochs,eta_min=args.task_learning_rate*1e-3,)

train_dataloader,test_dataloader,val_dataloader = build_dataloader(train_transform,0.7)

# print(len(train_dataloader),len(val_dataloader),len(test_dataloader))

def train(train_dataloader,val_dataloader):
    for epoch in range(args.task_epochs):
        # global_model.train()
        model.train()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        for images,labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features = global_model(images)

            proj,outputs = model(features)

        
            loss = criterion(outputs, labels)

            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()

            total_loss += loss.item() 
        avg_loss = total_loss / len(train_dataloader.dataset)

        task_scheduler.step()

        correct = 0
        model.eval()
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features = global_model(images)
                proj,outputs = model(features)

            val_output = outputs.argmax(dim=1)
            
            correct += (val_output == labels).sum().item()
        val_accuracy = correct / len(val_dataloader.dataset)

        dir = f'/users/student/pg/pg23/vaibhav.rathore/D_GCD/DG/project/checkpoints/{args.dataset_name}/bn2/'

        if not os.path.exists(dir):
            os.makedirs(dir)
        if epoch % 10 ==0:
            torch.save(model.state_dict(),dir+f"checkpoint_{args.dataset_name}_epoch{epoch}.pth")
        print(f"Epoch [{epoch+1}/{args.task_epochs}], Train_Loss: {avg_loss:.4f},Val_Accuracy: {val_accuracy*100:.4f}")


def test(test_dataloader,model_ckpt_path=None):
    if model_ckpt_path is not None:
        model.load_state_dict(torch.load(model_ckpt_path))
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            features = global_model(images)
            _,outputs = model(features)

            test_output = outputs.argmax(dim=1)
            correct += (test_output == labels).sum().item()
    test_accuracy = correct / len(test_dataloader.dataset)
    print(f"Test Accuracy: {test_accuracy*100:.4f}")



if __name__ == "__main__":

    if args.split == 'test':
        ckpt_path = os.listdir(f'/users/student/pg/pg23/vaibhav.rathore/D_GCD/DG/checkpoints/{args.dataset_name}/bn2')
        for i in ckpt_path:
            model_ckpt_path = f'/users/student/pg/pg23/vaibhav.rathore/D_GCD/DG/checkpoints/{args.dataset_name}/bn2/{i}'
            print(f"Evaluating model from checkpoint: {model_ckpt_path}")
            test(test_dataloader,model_ckpt_path=model_ckpt_path)
    else:
        train(train_dataloader,val_dataloader)

   