import math
import numpy as np
import torch
import helpers


def produce_partial_path(baseline, arr_3d, index, num_steps, batch_size):

    return np.concatenate(
        [
            np.expand_dims(
                baseline + (arr_3d - baseline) * index / (num_steps - 1),
                axis=0
            )
            for index in range(index*batch_size, (index+1)*batch_size)
        ]
    )


def integrated_gradients(image_tensor, model, target, num_steps=10, batch_size=10):

    device = image_tensor.device
    arr_4d = image_tensor.detach().cpu().numpy()
    
    baseline = np.zeros(arr_4d[0].shape, dtype='float32')
    integ_grads = np.zeros(arr_4d.shape, dtype='float32')
    predictions = np.zeros((arr_4d.shape[0],), dtype='float32')

    for num in range(arr_4d.shape[0]):
        for index in range(math.ceil(num_steps/batch_size)):
            
            input_arr_4d = produce_partial_path(baseline, arr_4d[num], index, num_steps, batch_size)
            grads, softmax_preds_2d = helpers.vanilla_gradients(torch.from_numpy(input_arr_4d).to(device), model, target)
            integ_grads[num] = np.add(integ_grads[num], np.sum(grads.detach().cpu().numpy(), axis=0))
        
        predictions[num] = softmax_preds_2d[-1].detach().cpu().numpy()
        integ_grads[num] = np.divide(integ_grads[num], num_steps)
        integ_grads[num] = np.multiply(integ_grads[num], arr_4d[num] - baseline)

    return torch.from_numpy(integ_grads).to(device), torch.from_numpy(predictions)


def get_integ_grad(image_tensor, model, target):
    
    integ_grads, prediction = integrated_gradients(image_tensor, model, target)
    
    norm_array_2d = helpers.grad_tensor_to_image_array(integ_grads[0])

    predictions = prediction.tolist()
    
    return norm_array_2d, predictions
