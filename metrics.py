import helpers
import evaluation
import vanilla_grad
import integ_grad
import smooth_grad
import rise_grad
import lime


def perform(grad_fn, image_tensor, model, target, csv_path) -> None:

    if grad_fn == 'vanilla':
        norm_array_2d, predictions = vanilla_grad.get_vanilla_grad(
            image_tensor, model, target)        
    
    elif grad_fn == 'integrated':
        norm_array_2d, predictions = integ_grad.get_integ_grad(
            image_tensor, model, target)

    elif grad_fn == 'smooth_vanilla':
        num_examples=20
        batch_size=10
        noise_perc=0.2
        norm_array_2d, predictions = smooth_grad.get_smooth_grad(
            num_examples, batch_size, noise_perc, 
            image_tensor, target, model, helpers.vanilla_gradients)
    
    elif grad_fn == 'smooth_integrated':
        num_examples=20
        batch_size=10
        noise_perc=0.2
        norm_array_2d, predictions = smooth_grad.get_smooth_grad(
            num_examples, batch_size, noise_perc, 
            image_tensor, target, model, integ_grad.integrated_gradients)
    
    elif grad_fn == 'rise_vanilla':
        num_examples = 100
        batch_size = 10
        small_dim = 8
        prob = 0.5
        noise_perc = 0.2
        norm_array_2d, predictions = rise_grad.get_rise_grad(
            num_examples, batch_size, small_dim, prob, noise_perc, 
            image_tensor, target, model, helpers.vanilla_gradients)
    
    elif grad_fn == 'rise_integrated':
        num_examples = 100
        batch_size = 10
        small_dim = 8
        prob = 0.5
        noise_perc = 0.2
        norm_array_2d, predictions = rise_grad.get_rise_grad(
            num_examples, batch_size, small_dim, prob, noise_perc, 
            image_tensor, target, model, integ_grad.integrated_gradients)
    
    elif grad_fn == 'lime':
        norm_array_2d = lime.get_lime(image_tensor, model)

    else:
        raise ValueError(f'grad_fn "{grad_fn}" not supported')
    
    helpers.visualize_sensitivity_map(norm_array_2d, in_notebook=False)

    deletion_scores = evaluation.deletion_scores(
        model, norm_array_2d, image_tensor, target,
        num_stages=100, batch_size=10, show_stages=True)
    
    evaluation.visualize_scores(deletion_scores, is_insert=False)

    delete_auc = evaluation.area_under_curve(deletion_scores)

    print('delete_auc', delete_auc)

    # with open(csv_path, 'a') as file:
    #     file.write(
    #         str(np.median(predictions)) + ',' +     # prd
    #         str(np.median(norm_array_2d)) + ',' +   # imp
    #         str(delete_auc) + '\n'                  # dlt
    #     )
