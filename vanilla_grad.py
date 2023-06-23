import helpers


def get_vanilla_grad(image_tensor, model, target):

    grads = helpers.vanilla_gradients(image_tensor, model, target)
    
    norm_array_2d = helpers.grad_tensor_to_image_array(grads[0,:,:,:])

    return norm_array_2d
