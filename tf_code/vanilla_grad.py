import numpy as np

import helpers


def get_vanilla_grad(image_tensor, target_size, model):
    '''
    Perform vanilla gradients algorithm and return sensitivity map and predictions.

    Parameters
        image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
        target_size: tuple of (int, int)
        model: trained model
    
    Returns
        norm_array_2d: array with 2 dimensions (rows, cols)
        predictions: list with prediction scores
    '''    
    arr_3d = helpers.load_image_to_3d_array(image_tensor, target_size)

    # add dimension because the model needs it
    # shape = (batch_size, rows, columns, channels)
    input_arr_4d = np.expand_dims(arr_3d, axis=0)    

    grads, softmax_preds_2d = helpers.vanilla_gradients(input_arr_4d, model)
    
    norm_array_2d = helpers.grad_tensor_to_image_array(grads[0,:,:,:])

    predictions = []
    predictions.extend(softmax_preds_2d.numpy())

    return norm_array_2d, predictions
