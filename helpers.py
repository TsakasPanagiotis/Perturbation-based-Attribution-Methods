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


def visualize_sensitivity_map(norm_array_2d, in_notebook):

    if in_notebook:
        return im.fromarray(norm_array_2d)
    else:
        img = im.fromarray(norm_array_2d)
        img.save('saliency.png')
