import helpers


def get_vanilla_grad(image_tensor, model, target):

    grads, softmax_preds_2d = helpers.vanilla_gradients(image_tensor, model, target)
    
    norm_array_2d = helpers.grad_tensor_to_image_array(grads[0,:,:,:])

    predictions = softmax_preds_2d.tolist()

    return norm_array_2d, predictions
