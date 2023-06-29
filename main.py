from train import train_model, evaluate_model
from utils import create_subset, preprocess_stl
from model import ResNet50
from data import PneumoniaTumorDataset_No,PneumoniaTumorDataset_Yes,MixedDataset
import argparse
from sklearn.decomposition import PCA
import torchvision
import torch
from tqdm import tqdm
import metrics 
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

        methods = ['lime', 'smooth_integrated', 'rise_integrated']
        print(f"Mode : {args.mode} using : {methods[0],methods[1],methods[2]}")
        images_dict = {}
        indexes_dict = {}
        avg_auc_dict = {}

        for method in methods:
            images_dict[method] = []
            indexes_dict[method] = { 'orig': {}, 'pert': {} }
            avg_auc_dict[method] = 0

        model.load_state_dict(torch.load('./resnet_' + str(args.experiment) + '_model.pt'))
        model = model.to(args.device)
        test_subset = create_subset(args,testset)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=2)
        print("The length of the subset for explaining is : ", len(test_subset))

        str_label_dict = {
            'stl': {
                0: 'airplane',
                1: 'bird',
                2: 'car',
                3: 'cat'
            },
            'tumor': {
                0: 'tumor',
                1: 'normal'
            },
            'pneumonia': {
                0: 'normal',
                1: 'pneumonia'
            }
        }
        assert args.experiment in str_label_dict.keys(), 'Invalid experiment'

        image_tensor: torch.Tensor
        i = 1

        for image_tensor, label in tqdm(test_loader):
            image_tensor = image_tensor.to(args.device)
            label = label.to(args.device)

            str_label = str_label_dict[args.experiment][label.item()]

            for method in methods: 
                saliency_map, images, indexes, avg_auc = metrics.perform(
                    method, deepcopy(image_tensor), model, label.item(), 
                    images_dict[method], indexes_dict[method], avg_auc_dict[method])
                
                images_dict[method] = images
                indexes_dict[method] = indexes
                avg_auc_dict[method] = avg_auc

                if i % 10 == 0:
                    print(f"Saving saliency map for step {i} for label : {str_label}")

                    inverse_transform = torchvision.transforms.Compose([
                        torchvision.transforms.Normalize([ 0., 0., 0. ], [ 1/0.229, 1/0.224, 1/0.225 ]),
                        torchvision.transforms.Normalize([ -0.485, -0.456, -0.406 ], [ 1., 1., 1. ]),
                        torchvision.transforms.ToPILImage()
                    ])
                    original_image = inverse_transform(image_tensor[0])
                    # original_image = original_image.resize((224,224), resample=Image.LANCZOS)
                    original_image.save(f'results_{args.seed}/{args.experiment}/original_image_{i}.png')                    
                    
                    img = Image.fromarray(saliency_map)
                    img = img.resize((224,224), resample=Image.LANCZOS)
                    img.save(f'results_{args.seed}/{args.experiment}/{method}_map_{i}.png')
            i += 1

        colors = ['blue', 'orange', 'green', 'purple', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan']
        markers = ['o', 'v', '^', '<', '>', 's', 'P', '*', 'X', 'D']

        for method in methods:
            avg_auc_dict[method] /= len(test_loader)
            print(method, 'avg_auc', avg_auc_dict[method])

            pca = PCA(n_components=2)
            pca_images = pca.fit_transform(images_dict[method])   

            for label in indexes_dict[method]['orig'].keys():
                
                plt.scatter(pca_images[indexes_dict[method]['orig'][label]][:,0], 
                           pca_images[indexes_dict[method]['orig'][label]][:,1],
                           c=colors[label], marker=markers[label], s=70,  label=f'orig {str_label}')
                
                plt.scatter(pca_images[indexes_dict[method]['pert'][label]][:,0], 
                            pca_images[indexes_dict[method]['pert'][label]][:,1],
                            c=colors[label], marker=markers[label], alpha=0.1, s=30, label=f'pert {str_label}')

            ax = plt.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.title(f'PCA for {method}')
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            plt.savefig(f'results_{args.seed}/{args.experiment}/{method}_pca.png', facecolor='w', bbox_inches='tight')
            plt.close()


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
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    main(args)