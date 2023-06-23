import numpy as np

import helpers
import evaluation
import vanilla_grad
import integ_grad
import smooth_grad
import rise_grad


def perform(grad_fn, image_tensor, target_size, model, csv_path) -> None:
    '''
    Write metrics of image to corresponding csv.

    Parameters
        grad_fn: method that calculates the gradients
        image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
        target_size: tuple of (int, int)
        model: the model that makes the predictions
        csv_path: path of csv with results for the grad_fn
    '''

    if grad_fn == 'vanilla':
        norm_array_2d, predictions = vanilla_grad.get_vanilla_grad(
            image_tensor, target_size, model)
    
    elif grad_fn == 'integrated':
        norm_array_2d, predictions = integ_grad.get_integ_grad(
            image_tensor, target_size, model)

    elif grad_fn == 'smooth_vanilla':
        num_examples=20
        batch_size=10
        noise_perc=0.2
        norm_array_2d, predictions = smooth_grad.get_smooth_grad(
            num_examples, batch_size, noise_perc, 
            image_tensor, target_size, model, helpers.vanilla_gradients)
    
    elif grad_fn == 'smooth_integrated':
        num_examples=20
        batch_size=10
        noise_perc=0.2
        norm_array_2d, predictions = smooth_grad.get_smooth_grad(
            num_examples, batch_size, noise_perc, 
            image_tensor, target_size, model, integ_grad.integrated_gradients)
    
    elif grad_fn == 'rise_vanilla':
        num_examples = 100
        batch_size = 10
        small_dim = 8
        prob = 0.5
        noise_perc = 0.2
        norm_array_2d, predictions = rise_grad.get_rise_grad(
            num_examples, batch_size, small_dim, prob, noise_perc, 
            image_tensor, target_size, model, helpers.vanilla_gradients)
    
    elif grad_fn == 'rise_integrated':
        num_examples = 100
        batch_size = 10
        small_dim = 8
        prob = 0.5
        noise_perc = 0.2
        norm_array_2d, predictions = rise_grad.get_rise_grad(
            num_examples, batch_size, small_dim, prob, noise_perc, 
            image_tensor, target_size, model, integ_grad.integrated_gradients)

    else:
        raise ValueError("only ('', 'smooth_', 'rise_') X ('vanilla', 'integrated') combinations supported")

    deletion_scores = evaluation.deletion_scores(
        model, norm_array_2d, image_tensor, target_size, 
        num_stages=100, batch_size=10, show_stages=False)

    delete_auc = evaluation.area_under_curve(deletion_scores)

    with open(csv_path, 'a') as file:
        file.write(
            str(np.median(predictions)) + ',' +     # prd
            str(np.median(norm_array_2d)) + ',' +   # imp
            str(delete_auc) + '\n'                  # dlt
        )
