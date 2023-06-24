import numpy as np
from skimage.transform import resize
import torch
import helpers


def generate_masks(batch_size, small_dim, prob, target_size):
    
    cell_size = np.ceil(np.array(target_size) / small_dim)
    up_size = (small_dim + 1) * cell_size

    grid = np.random.rand(batch_size, small_dim, small_dim) < prob
    grid = grid.astype('float32')

    masks = np.empty((batch_size, *target_size), dtype='float32')

    for i in range(batch_size):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(
            grid[i], 
            up_size, 
            order=1, 
            mode='reflect',
            anti_aliasing=False)[x:x + target_size[0], y:y + target_size[1]]
    
    masks = masks.reshape(-1, 1, *target_size)

    return masks


def produce_noisy_masks(masks, batch_size, noise_perc, arr_3d):
    
    # new shape with batch_size dimension
    new_shape = (batch_size,) + arr_3d.shape
    
    # make normal noise
    # shape : (batch_size, rows=224, columns=224, channels=3)
    # dtype : float32
    noise = np.random.normal(
        loc=0, # mean
        scale=noise_perc * (np.max(arr_3d) - np.min(arr_3d)), # stdev
        size=new_shape
    ).astype('float32')

    # multiply masks with noise element-wise
    # broadcasting : masks : batch_size x 224 x 224 x 1
    #                noise : batch_size x 224 x 224 x 3
    #       noisy_masks_4d : batch_size x 224 x 224 x 3
    # dtype : float32
    noisy_masks_4d = noise * (1 - masks)

    return noisy_masks_4d


def get_weighted_grad(grads, softmax_preds_2d, batch_size):

    softmax_preds_4d = torch.reshape(softmax_preds_2d, (batch_size,1,1,1))

    mul_grad = grads * softmax_preds_4d

    weighted_grad = torch.sum(mul_grad, axis=0)

    return weighted_grad


def get_rise_grad(num_examples, batch_size, small_dim, prob, noise_perc, image_tensor, target, model, grad_func):
    
    rise_grad = torch.zeros(image_tensor[0].shape, dtype=torch.float32).to(image_tensor.device)
    predictions = []

    for _ in range(np.math.ceil(num_examples/batch_size)):
        # prepare input
        masks = generate_masks(batch_size, small_dim, prob, target_size=image_tensor.shape[2:])
        noisy_masks_4d = produce_noisy_masks(masks, batch_size, noise_perc, image_tensor[0].detach().cpu().numpy())
        input_arr_4d = noisy_masks_4d + np.broadcast_to(image_tensor[0].detach().cpu().numpy(), noisy_masks_4d.shape)
        # prepare output
        grads, softmax_preds_2d = grad_func(torch.from_numpy(input_arr_4d).to(image_tensor.device), model, target)
        predictions.extend(softmax_preds_2d.tolist())
        weighted_grad = get_weighted_grad(grads, softmax_preds_2d.to(image_tensor.device), batch_size)
        rise_grad = rise_grad + weighted_grad
    
    # get weighted average gradient as RISE paper says
    rise_grad = rise_grad / (num_examples * prob)
    norm_array_2d = helpers.grad_tensor_to_image_array(rise_grad)

    return norm_array_2d, predictions
