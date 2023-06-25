import torch
import helpers
import numpy as np
from copy import deepcopy


def produce_noisy_images(batch_size, image_tensor, noise_perc):

    new_shape = (batch_size,) + image_tensor[0].shape

    noise = torch.normal(
        mean=0.0,
        std=noise_perc * (image_tensor.max() - image_tensor.min()),
        size=new_shape
    ).to(torch.float32).to(image_tensor.device)

    repeat_image_tensor = image_tensor.repeat(batch_size,1,1,1)
    
    noisy_arr_4d = repeat_image_tensor + noise

    return noisy_arr_4d


def get_smooth_grad(num_examples, batch_size, noise_perc, image_tensor, target, model, grad_func, images, indexes):

    images.append(image_tensor[0].detach().cpu().numpy().flatten())
    if target not in indexes['orig'].keys():
        indexes['orig'][target] = []
    indexes['orig'][target].append(len(images)-1)

    smooth_grad = torch.zeros(image_tensor[0].shape, dtype=torch.float32).to(image_tensor.device)
    predictions = []

    for _ in range(np.math.ceil(num_examples/batch_size)):

        input_arr_4d = produce_noisy_images(batch_size, image_tensor, noise_perc)

        for image in deepcopy(input_arr_4d):
            images.append(image.detach().cpu().numpy().flatten())
            if target not in indexes['pert'].keys():
                indexes['pert'][target] = []
            indexes['pert'][target].append(len(images)-1)

        grads, softmax_preds_2d = grad_func(input_arr_4d, model, target)

        predictions.extend(softmax_preds_2d.tolist())
        smooth_grad = smooth_grad + torch.sum(grads, dim=0)
    
    smooth_grad = smooth_grad / num_examples
    norm_array_2d = helpers.grad_tensor_to_image_array(smooth_grad)

    return norm_array_2d, predictions, images, indexes
