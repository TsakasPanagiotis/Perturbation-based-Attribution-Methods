import torch
from PIL import Image as im


def grad_tensor_to_image_array(grad):

    grayscale_tensor = torch.max(torch.abs(grad), dim=0)[0]

    scaled_tensor = (grayscale_tensor - grayscale_tensor.min()) / (grayscale_tensor.max() - grayscale_tensor.min())

    normalized_tensor = (255 * scaled_tensor).to(torch.uint8)

    norm_array_2d = normalized_tensor.detach().cpu().numpy()

    return norm_array_2d


def vanilla_gradients(input_array: torch.Tensor, model, target):
    
    input_array.requires_grad = True
    
    logits = model(input_array)

    target_logits = logits[:,target]

    target_logits.backward(gradient=torch.tensor([1.]*target_logits.shape[0]).to(input_array.device))
    
    grads = input_array.grad

    softmax_preds_2d = torch.exp(target_logits) / torch.sum(torch.exp(logits), dim=1)

    return grads, softmax_preds_2d


# def get_info(name):
#     '''
#     Get image path and index of given class.

#     Parameters
#         name: str
    
#     Returns
#         img_path: str
#         class_index: int
#     '''
#     img_folder = 'images/'

#     if name in ['elephant', 'tusker']:
#         return img_folder + 'elephant.jpg', 101
    
#     elif name in ['cat', 'egyptian cat', 'egyptian_cat']:
#         return img_folder + 'cat.jpg', 285

#     elif name in ['ostrich']:
#         return img_folder + 'ostrich.jpg', 9
    
#     elif name in ['fox', 'red fox', 'red_fox']:
#         return img_folder + 'red_fox.jpg', 277
    
#     elif name in ['goldfish']:
#         return img_folder + 'goldfish.jpg', 1

#     elif name in ['lizard', 'green lizard', 'green_lizard']:
#         return img_folder + 'green_lizard.jpg', 40
    
#     elif name in ['gazelle']:
#         return img_folder + 'gazelle.jpg', 353
    
#     elif name in ['dog', 'terrier', 'norwich terrier', 'norwich_terrier']:
#         return img_folder + 'terrier.jpg', 186


def visualize_sensitivity_map(norm_array_2d, in_notebook):

    if in_notebook:
        return im.fromarray(norm_array_2d)
    else:
        img = im.fromarray(norm_array_2d)
        img.save('saliency.png')


# def visualize_image(image_tensor, target_size, in_notebook):
#     '''
#     Visualize image given by name at given size.

#     Parameters
#         image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
#         target_size: tuple of (int, int)
#         in_notebook: boolean that shows image in notebook or opens Photos
#     '''
#     # img_path, _ = get_info(name)
#     arr_3d = load_image_to_3d_array(image_tensor, target_size)
#     if in_notebook:
#         return image.array_to_img(arr_3d)
#     else:
#         image.array_to_img(arr_3d).show()
