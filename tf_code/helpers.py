import tensorflow as tf
from PIL import Image as im
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def load_vgg16(activation):
    '''
    Load VGG16 model with pretrained imagenet weights with or without softmax after output.

    Parameters
        activation: str either 'softmax' or 'linear'
    
    Returns
        model: VGG16 model
    '''
    if activation == 'softmax':
        model = VGG16(weights='imagenet', include_top=True, classifier_activation='softmax')
    
    elif activation == 'linear':
        model = VGG16(weights='imagenet', include_top=True, classifier_activation=None)
    
    else:
        model = None
    
    return model


def load_image_to_3d_array(image_tensor, target_size):
    '''
    Load image from path to array.

    Parameters
        image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
        target_size: tuple of (int, int)
    
    Returns
        arr_3d: array with 3 dimensions (rows, cols, channels)
    '''
    # load image to the size the model needs
    # img = image.load_img(img_path, target_size=target_size)
    img = im.fromarray(tf.image.resize(image_tensor[0], target_size).numpy().astype('uint8'))
    
    # convert image to array
    # shape : (rows=224, columns=224, channels=3)
    # dtype : float32
    arr_3d = image.img_to_array(img)

    return arr_3d


def prepare_4d_array_to_tensor(arr_4d):
    '''
    Prepare input tensor from array as VGG16 needs it.

    Parameters
        arr_4d: array with 4 dimensions (batch_size, rows, columns, channels)
    
    Returns
        input_4d_tensor: tensor with 4 dimensions (batch_size, rows, columns, channels)
    '''
    # preprocess final array with model function
    # shape : (batch_size=num_examples, rows=224, columns=224, channels=3)
    # dtype : float32
    input_4d_arr = preprocess_input(arr_4d.copy())
    
    # convert array to tensor because GradientTape needs it
    # shape : (batch_size=num_examples, rows=224, columns=224, channels=3)
    # dtype : float32
    input_4d_tensor = tf.Variable(input_4d_arr)

    return input_4d_tensor


def grad_tensor_to_image_array(grad):
    '''
    Convert gradients tensor to grayscale array.

    Parameters
        grad: tensor with 3 dimensions (rows, cols, channels)
    
    Returns
        norm_array_2d: array with 3 dimensions (rows, cols, channels=1)
    '''
    # reduce channels dimension keeping max of absolute channel gradients
    # shape : (rows=224, columns=224)
    # dtype : float32
    grayscale_tensor = tf.reduce_max(tf.abs(grad), axis=-1)

    # scale to 0-1 : (value − min_value) / (max_value − min_value)
    # shape : (rows=224, columns=224, channels=1)
    # dtype : float32
    scaled_tensor = tf.math.divide(
        tf.math.subtract(
            grayscale_tensor,
            tf.reduce_min(grayscale_tensor)
        ),
        tf.math.subtract(
            tf.reduce_max(grayscale_tensor),
            tf.reduce_min(grayscale_tensor)
        )
    )

    # scale tensor as greyscale image to 0-255
    # shape : (rows=224, columns=224, channels=1)
    # dtype : uint8
    normalized_tensor = tf.cast(scaled_tensor * 255, tf.uint8)

    # convert 2d tensor to 2d array for visualization
    # shape : (rows=224, columns=224, channels=1)
    # dtype : uint8
    norm_array_2d = normalized_tensor.numpy()

    return norm_array_2d


def vanilla_gradients(input_array, model):
    '''
    Perform vanilla gradients and return gradients with softmax scores.

    Parameters
        input_array: array with 4 dimensions (batch_size, rows, cols, channels)
        model: VGG16 without softmax
    
    Returns
        grads: tensor with 4 dimensions (batch_size, rows, cols, channels)
        softmax_preds_2d: tensor with 2 dimensions (batch_size,)
    '''

    input_tensor = prepare_4d_array_to_tensor(input_array)

    # run the model while watching the input tensor
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        
        # shape : (batch_size=num_examples, num_classes=1000)
        predictions = model(input_tensor)
        
        # shape : (batch_size=num_examples,)
        scores_for_class = tf.math.reduce_max(predictions, axis=1)
    
    # get the class scores gradient tensor wrt the input tensor
    # shape : (batch_size=num_examples, rows=224, columns=224, channels=3)
    # dtype : float32
    grads = tape.gradient(scores_for_class, input_tensor)
    
    # convert linear predictions to softmax scores
    # shape : (batch_size=num_examples,)
    softmax_preds_2d = tf.math.divide(
        tf.math.exp(scores_for_class),
        tf.math.reduce_sum(tf.math.exp(predictions), axis=1)
    )

    return grads, softmax_preds_2d


def get_info(name):
    '''
    Get image path and index of given class.

    Parameters
        name: str
    
    Returns
        img_path: str
        class_index: int
    '''
    img_folder = 'images/'

    if name in ['elephant', 'tusker']:
        return img_folder + 'elephant.jpg', 101
    
    elif name in ['cat', 'egyptian cat', 'egyptian_cat']:
        return img_folder + 'cat.jpg', 285

    elif name in ['ostrich']:
        return img_folder + 'ostrich.jpg', 9
    
    elif name in ['fox', 'red fox', 'red_fox']:
        return img_folder + 'red_fox.jpg', 277
    
    elif name in ['goldfish']:
        return img_folder + 'goldfish.jpg', 1

    elif name in ['lizard', 'green lizard', 'green_lizard']:
        return img_folder + 'green_lizard.jpg', 40
    
    elif name in ['gazelle']:
        return img_folder + 'gazelle.jpg', 353
    
    elif name in ['dog', 'terrier', 'norwich terrier', 'norwich_terrier']:
        return img_folder + 'terrier.jpg', 186


def visualize_sensitivity_map(norm_array_2d, in_notebook):
    '''
    Show the image of the sensitivity map.

    Parameters
        norm_array_2d: array with 2 dimensions (rows, cols)
        in_notebook: boolean that shows image in notebook or opens Photos
    '''
    if in_notebook:
        return im.fromarray(norm_array_2d)
    else:
        im.fromarray(norm_array_2d).show()


def visualize_image(image_tensor, target_size, in_notebook):
    '''
    Visualize image given by name at given size.

    Parameters
        image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
        target_size: tuple of (int, int)
        in_notebook: boolean that shows image in notebook or opens Photos
    '''
    # img_path, _ = get_info(name)
    arr_3d = load_image_to_3d_array(image_tensor, target_size)
    if in_notebook:
        return image.array_to_img(arr_3d)
    else:
        image.array_to_img(arr_3d).show()
