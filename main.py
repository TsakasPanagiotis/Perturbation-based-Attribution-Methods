import argparse
from train import train_model, evaluate_model
from model import VGG16
from data import PneumoniaTumorDataset_No,PneumoniaTumorDataset_Yes,MixedDataset
import torchvision
import torch

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
        model = VGG16(2)
    
    elif args.experiment == 'tumor':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = (256,256)),
            torchvision.transforms.RandomRotation(degrees = (-20,+20)),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
            
        trainset = torchvision.datasets.ImageFolder('./brain-tumor-classification-mri/Training/', transform=transform)
        testset = torchvision.datasets.ImageFolder('./brain-tumor-classification-mri/Testing/', transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        model = VGG16(4)

    model = model.to(args.device)
    train_model(args, model, trainloader)
    test_acc = evaluate_model(args, model, testloader)
    print(f"Test accuracy on {len(testloader)} samples : {test_acc}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='pneumonia', type=str,
                        help='Choose "pneumonia" or "tumor"')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose "cpu" or "cuda"')
    # parser.add_argument('--num_images_per_class', default=25, type=int,
    #                     help='Images for each class for the subset')
    args = parser.parse_args()
    main(args)