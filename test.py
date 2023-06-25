import torch
import torchvision
from torch.utils.data import DataLoader
# import helpers
import metrics
from copy import deepcopy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.vgg16(pretrained=True)
model = model.to(device)
model.eval()

transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)


image_tensor: torch.Tensor

image_tensor, label = next(iter(test_loader)) # torch.Size([1, 1, 224, 224]) torch.float32
image_tensor = image_tensor.to(device)
label = label.to(device)

# # lines for mnist
# image_tensor = image_tensor.permute(0,2,3,1) # torch.Size([1, 224, 224, 1]) torch.float32
# image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=-1) # torch.Size([1, 224, 224, 3]) torch.float32
# image_tensor = image_tensor.permute(0,3,1,2) # torch.Size([1, 3, 224, 224]) torch.float32


# metrics.perform('vanilla', deepcopy(image_tensor), model, label.item(), 'vanilla_results.csv')
# metrics.perform('integrated', deepcopy(image_tensor), model, label.item(), 'integrated_results.csv')
# metrics.perform('smooth_vanilla', deepcopy(image_tensor), model, label.item(), 'smooth_vanilla_results.csv')
# metrics.perform('smooth_integrated', deepcopy(image_tensor), model, label.item(), 'smooth_integrated_results.csv')
# metrics.perform('rise_vanilla', deepcopy(image_tensor), model, label.item(), 'rise_vanilla_results.csv')
# metrics.perform('rise_integrated', deepcopy(image_tensor), model, label.item(), 'rise_integrated_results.csv')

metrics.perform('lime', deepcopy(image_tensor), model, label.item(), 'lime_results.csv')
