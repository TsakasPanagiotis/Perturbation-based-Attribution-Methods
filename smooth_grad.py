import numpy as np
import tensorflow as tf

import helpers


def produce_noisy_images(batch_size, arr_3d, noise_perc):
    '''
    Produce noisy instances of image.

    Parameters
        batch_size: int
        arr_3d: array with 3 dimensions (rows, cols, channels)
        noise_perc: float in range [0,1]
    
    Returns
        noisy_arr_4d: array with 4 dimensions (batch_size, rows, cols, channels)
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
    
    # repeat image array for num_examples times
    # shape : (batch_size, rows=224, columns=224, channels=3)
    # dtype : float32
    repeat_arr_3d = np.broadcast_to(arr_3d, new_shape)
    
    # add noise to image array
    # shape : (batch_size, rows=224, columns=224, channels=3)
    # dtype : float32
    noisy_arr_4d = repeat_arr_3d + noise

    return noisy_arr_4d


def get_smooth_grad(num_examples, batch_size, noise_perc, image_tensor, target_size, model, grad_func):
    '''
    Perform smooth grad algorithm and return sensitivity map.

    Parameters
        num_examples: int
        batch_size: int
        noise_perc: float
        image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
        target_size: typle of (int, int)
        model: trained model
        grad_func: method to get gradients
    
    Returns
        norm_array_2d: array with 2 dimensions (rows, cols)
        predictions: list of softmax scores for each noisy instance
    '''   

    arr_3d = helpers.load_image_to_3d_array(image_tensor, target_size)
    smooth_grad = tf.zeros(arr_3d.shape, dtype=tf.dtypes.float32)
    predictions = []

    for _ in range(np.math.ceil(num_examples/batch_size)):

        # prepare input
        input_arr_4d = produce_noisy_images(batch_size, arr_3d, noise_perc)
        
        # prepare output
        grads, softmax_preds_2d = grad_func(input_arr_4d, model)
        predictions.extend(softmax_preds_2d.numpy())
        smooth_grad = tf.math.add(smooth_grad, tf.math.reduce_sum(grads, axis=0))
    
    smooth_grad = tf.math.divide(smooth_grad, num_examples)
    norm_array_2d = helpers.grad_tensor_to_image_array(smooth_grad)

    return norm_array_2d, predictions
