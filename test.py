import torch
import torchvision
from torch.utils.data import DataLoader
import metrics
from copy import deepcopy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.vgg16(pretrained=True)
model = model.to(device)
model.eval()

transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)


# image_tensor: torch.Tensor

# image_tensor, label = next(iter(test_loader)) # torch.Size([1, 1, 224, 224]) torch.float32
# image_tensor = image_tensor.to(device)
# label = label.to(device)

# # lines for mnist
# image_tensor = image_tensor.permute(0,2,3,1) # torch.Size([1, 224, 224, 1]) torch.float32
# image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=-1) # torch.Size([1, 224, 224, 3]) torch.float32
# image_tensor = image_tensor.permute(0,3,1,2) # torch.Size([1, 3, 224, 224]) torch.float32


methods = ['lime', 'smooth_integrated', 'rise_integrated']
        
images_dict = {}
indexes_dict = {}
avg_auc_dict = {}

for method in methods:
    images_dict[method] = []
    indexes_dict[method] = { 'orig': {}, 'pert': {} }
    avg_auc_dict[method] = 0

i = 1
counter = 0
for image_tensor, label in test_loader:
    image_tensor = image_tensor.to(device)
    label = label.to(device)

    for method in methods:

        saliency_map, images, indexes, avg_auc = metrics.perform(
            method, deepcopy(image_tensor), model, label.item(), 
            images_dict[method], indexes_dict[method], avg_auc_dict[method])
        
        images_dict[method] = images
        indexes_dict[method] = indexes
        avg_auc_dict[method] = avg_auc

        if i % 5 == 0:
            print(f"Saving saliency map for {method} step {i} - label {label}")
            img = Image.fromarray(saliency_map)
            img = img.resize((224,224), resample=Image.LANCZOS)
            img.save(f'{method}_{i}.png')

    i += 1
    counter += 1
    if counter == 5:
        break


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
                    c=colors[label], marker=markers[label], s=70,  label=f'orig {label}')
        
        plt.scatter(pca_images[indexes_dict[method]['pert'][label]][:,0], 
                    pca_images[indexes_dict[method]['pert'][label]][:,1],
                    c=colors[label], marker=markers[label], alpha=0.1, s=20, label=f'pert {label}')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.title(f'PCA for {method}')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.savefig(f'{method}_distr.png', facecolor='w', bbox_inches='tight')
    plt.close()
