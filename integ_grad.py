import math

import numpy as np
import tensorflow as tf

import helpers


def produce_partial_path(baseline, arr_3d, index, num_steps, batch_size):
    '''
    Produce partial path of images according to integrated gradients formula.

    Parameters
        baseline: base image, array with 3 dimesnions
        arr_3d: array with 3 dimensions (rows, cols, channels)
        index: int
        num_steps: int for steps of partial path
        batch_size: int

    Returns
        array with 4 dimensions (num_steps, rows, cols, channels)

    '''
    return np.concatenate(
        [
            np.expand_dims(
                baseline + (arr_3d - baseline) * index / (num_steps - 1),
                axis=0
            )
            for index in range(index*batch_size, (index+1)*batch_size)
        ]
    )


def integrated_gradients(arr_4d, model, num_steps=10, batch_size=10):
    '''
    Perform integrated gradients and return gradients with softmax scores.

    Parameters
        input_array: array with 4 dimensions (batch_size, rows, cols, channels)
        model: VGG16 without softmax
        num_steps: int for steps of partial path
        batch_size: int
    
    Returns
        integ_grads: tensor with 4 dimensions (batch_size, rows, cols, channels)
        softmax_preds_2d: tensor with 2 dimensions (batch_size,)
    '''
    
    baseline = np.zeros(arr_4d[0].shape, dtype='float32')
    integ_grads = np.zeros(arr_4d.shape, dtype='float32')
    predictions = np.zeros((arr_4d.shape[0],), dtype='float32')

    for num in range(arr_4d.shape[0]):

        for index in range(math.ceil(num_steps/batch_size)):
            
            # prepare input
            input_arr_4d = produce_partial_path(baseline, arr_4d[num], index, num_steps, batch_size)
            # prepare output
            grads, softmax_preds_2d = helpers.vanilla_gradients(input_arr_4d, model)
            integ_grads[num] = np.add(integ_grads[num], np.sum(grads.numpy(), axis=0))
        
        predictions[num] = softmax_preds_2d[-1].numpy()
        integ_grads[num] = np.divide(integ_grads[num], num_steps)
        integ_grads[num] = np.multiply(integ_grads[num], arr_4d[num] - baseline)

    return tf.convert_to_tensor(integ_grads), tf.convert_to_tensor(predictions)


def get_integ_grad(image_tensor, target_size, model):
    '''
    Perform integrated gradients algorithm and return sensitivity map and predictions.

    Parameters
        image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
        target_size: tuple of (int, int)
        model: trained model
    
    Returns
        norm_array_2d: array with 2 dimensions (rows, cols)
        predictions: list with prediction scores
    '''
    
    arr_3d = helpers.load_image_to_3d_array(image_tensor, target_size)
    
    integ_grads, prediction = integrated_gradients(np.expand_dims(arr_3d, axis=0), model)
    
    norm_array_2d = helpers.grad_tensor_to_image_array(integ_grads[0])

    predictions = []
    predictions.extend(prediction.numpy())
    
    return norm_array_2d, predictions
