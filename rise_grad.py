import numpy as np
import tensorflow as tf
from skimage.transform import resize

import helpers


def generate_masks(batch_size, small_dim, prob, target_size):
    '''
    Returns batch_size number of masks.

    Parameters:
        batch_size: batch size to process the copies
        small_dim: mask size parameter
        prob: probability of masking a pixel
        target_size: the required model input size

    Returns:
        masks: zero to one range masks
    '''
    
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
    
    masks = masks.reshape(-1, *target_size, 1) # shape : (batch_size, 224, 224, 1)

    return masks


def produce_noisy_masks(masks, batch_size, noise_perc, arr_3d):
    '''
    Returns masks with noise to later make copies of the original image.

    Parameters:
        masks: zero to one range masks
        batch_size: batch size to process the copies
        noise_perc: amount of noise to add
        arr_3d: array of the original image

    Returns:
        noisy_masks_4d: masks with noise
    '''
    
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
    '''
    Returns the weighted sum of the gradients and the prediction scores.

    Parameters:
        grads: tensor of gradients
        softmax_preds_2d: prediction scores
        batch_size: batch size to process the copies

    Returns:
        weighted_grad: weighted sum of the gradients and the prediction scores
    '''

    # reshape softmax scores
    softmax_preds_4d = tf.reshape(softmax_preds_2d, (batch_size,1,1,1))
    
    # multiply softmax scores with corresponding gradient tensor
    # shape : (num_examples, 224, 224, 3)
    mul_grad = tf.math.multiply(grads, softmax_preds_4d)
    
    # sum gradients across examples
    # shape : (224, 224, 3)
    weighted_grad = tf.math.reduce_sum(mul_grad, axis=0)

    return weighted_grad


def get_rise_grad(num_examples, batch_size, small_dim, prob, noise_perc, image_tensor, target_size, model, grad_func):
    '''
    Returns the sensitivity map of the given image and also the prediction scores of the copies that get masked with noise.

    Parameters:
        num_examples: numbers of copies to make
        batch_size: batch size to process the copies
        small_dim: mask size parameter
        prob: probability of masking a pixel
        noise_perc: amount of noise to add
        image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
        target_size: the required model input size
        model: the model that makes the predictions
        grad_func: method that calculates the gradients

    Returns:
        norm_array_2d: the final sensitivity map
        predictions: list of prediction scores of the copies
    '''
    
    arr_3d = helpers.load_image_to_3d_array(image_tensor, target_size)
    rise_grad = tf.zeros(arr_3d.shape, dtype=tf.dtypes.float32)
    predictions = []

    for _ in range(np.math.ceil(num_examples/batch_size)):
        # prepare input
        masks = generate_masks(batch_size, small_dim, prob, target_size)
        noisy_masks_4d = produce_noisy_masks(masks, batch_size, noise_perc, arr_3d)
        input_arr_4d = noisy_masks_4d + np.broadcast_to(arr_3d, noisy_masks_4d.shape)
        # prepare output
        grads, softmax_preds_2d = grad_func(input_arr_4d, model)
        predictions.extend(softmax_preds_2d.numpy())
        weighted_grad = get_weighted_grad(grads, softmax_preds_2d, batch_size)
        rise_grad = tf.math.add(rise_grad, weighted_grad)
    
    # get weighted average gradient as RISE paper says
    rise_grad = tf.math.divide(rise_grad, num_examples * prob)
    norm_array_2d = helpers.grad_tensor_to_image_array(rise_grad)

    return norm_array_2d, predictions
