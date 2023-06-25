from train import train_model, evaluate_model
from utils import create_subset, preprocess_stl
from model import ResNet50
from data import PneumoniaTumorDataset_No,PneumoniaTumorDataset_Yes,MixedDataset
import argparse
import torchvision
import torch
from tqdm import tqdm
import metrics 
from copy import deepcopy
from PIL import Image


def main(args):

    if args.experiment == 'pneumonia':

        no_train_dataset = PneumoniaTumorDataset_No('./chest_xray/train/NORMAL')
        yes_train_dataset = PneumoniaTumorDataset_Yes('./chest_xray/train/PNEUMONIA')
        trainset = MixedDataset(yes_train_dataset, no_train_dataset)
        no_test_dataset = PneumoniaTumorDataset_No('./chest_xray/test/NORMAL')
        yes_test_dataset = PneumoniaTumorDataset_Yes('./chest_xray/test/PNEUMONIA')
        testset = MixedDataset(yes_test_dataset, no_test_dataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        model = ResNet50(2)
    
    elif args.experiment == 'tumor':

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = (256,256)),
            torchvision.transforms.RandomRotation(degrees = (-20,+20)),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = (256,256)),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        
        trainset = torchvision.datasets.ImageFolder('./brain-tumor-classification/Training', transform=train_transform)
        testset = torchvision.datasets.ImageFolder('./brain-tumor-classification/Testing', transform=test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        model = ResNet50(2)

    
    elif args.experiment == 'stl':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128), Image.LANCZOS),
            #torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        original_set = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
        trainset, testset = preprocess_stl(original_set, [4,5,6,7,8,9])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        model = ResNet50(4)


    if args.mode == 'train':
        model = model.to(args.device)
        train_model(args, model, trainloader)
        test_acc = evaluate_model(args, model, testloader)
        print(f"Test accuracy on {len(testset)} samples : {test_acc}")
    
    elif args.mode == 'explain':

        model.load_state_dict(torch.load('./resnet_' + str(args.experiment) + '_model.pt'))
        model = model.to(args.device)
        test_subset = create_subset(args,testset)
        test_loader =  torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=2)

        test_acc = evaluate_model(args, model, testloader)
        print(f"Test accuracy on {len(testset)} samples : {test_acc}")

        image_tensor: torch.Tensor
        i = 1
        for image_tensor, label in tqdm(test_loader):
            print(i)
            image_tensor = image_tensor.to(args.device)
            label = label.to(args.device)
            metrics.perform('vanilla', deepcopy(image_tensor), model, label.item(), 'vanilla_results.csv')
            metrics.perform('integrated', deepcopy(image_tensor), model, label.item(), 'integrated_results.csv')
            metrics.perform('smooth_vanilla', deepcopy(image_tensor), model, label.item(), 'smooth_vanilla_results.csv')
            metrics.perform('smooth_integrated', deepcopy(image_tensor), model, label.item(), 'smooth_integrated_results.csv')
            metrics.perform('rise_vanilla', deepcopy(image_tensor), model, label.item(), 'rise_vanilla_results.csv')
            metrics.perform('rise_integrated', deepcopy(image_tensor), model, label.item(), 'rise_integrated_results.csv')
            i += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='pneumonia', type=str,
                        help='Choose "pneumonia" or "tumor"')
    parser.add_argument('--mode', default='explain', type=str,
                        help='Choose "train" or "explain"')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate for training.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose "cpu" or "cuda"')
    args = parser.parse_args()
    main(args)