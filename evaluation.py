import helpers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def histogram_of_predictions(predictions, range=(0,1), bins=100):
    '''
    Show histogram of prediction scores.

    Parameters    
        predictions: list of floats
        range: tuple of (int, int)        
        bins: int
    '''
    plt.hist(predictions, range=range, bins=bins)
    plt.xlabel('πιθανότητα πρόβλεψης')
    plt.ylabel('αριθμός αντιγράφων με θόρυβο')
    plt.xlim([0,1])
    plt.show() 


def histogram_of_sensitivity_map(norm_array_2d, range=(0,255), bins=255):
    '''
    Show histogram of sensitivity map values.

    Parameters        
        norm_array_2d: array with 2 dimensions (rows, cols)
        range: tuple of (int, int)        
        bins: int
    '''
    plt.hist(norm_array_2d.flatten().tolist(), range=range, bins=bins)
    plt.xlabel('τιμή σπουδαιότητας')
    plt.ylabel('αριθμός εικονοστοιχείων')
    plt.xlim([0,255])
    plt.show() 


def get_preds_array(model, input_arr_4d):
    '''
    Get softmax predictions for input array batch.

    Parameters    
        model: trained model
        input_arr_4d: array with 4 dimensions (batch_size, rows, cols, channels)
    
    Returns
        softmax_preds_2d.numpy(): array with 2 dimensions (batch_size, 1)
    '''
    input_tensor_4d = helpers.prepare_4d_array_to_tensor(input_arr_4d)
    predictions = model(input_tensor_4d)
    
    softmax_preds_2d = tf.math.divide(
        tf.math.exp(tf.math.reduce_max(predictions, axis=1)),
        tf.math.reduce_sum(tf.math.exp(predictions), axis=1)
    )
    
    return softmax_preds_2d.numpy()


def get_deletion_stages_batch(arr_3d, importances, num_stages, batch_size):
    '''
    Get batch_size stages of deletion for current importances.

    Parameters    
        arr_3d: array with 3 dimensions (rows, cols, channels)
        importances: array with 2 dimensions (rows, cols)        
        num_stages: int        
        batch_size: int

    Returns
        stages: array with 4 dimensions (batch_size, rows, cols, channels)
    '''
    stages = np.zeros( shape=(batch_size,)+arr_3d.shape, dtype='float32')
    for stage in range(batch_size):
        for _ in range(50000//num_stages):
            ind = np.unravel_index(np.argmax(importances, axis=None), importances.shape)
            importances[ind] = 0
            arr_3d[ind] = np.array([0,0,0], dtype='float32')
        stages[stage] = arr_3d.copy()
    
    return stages


def visualize_stages(stages):
    '''
    Visualize grid of deletion stages.

    Parameters
        stages: array with 4 dimensions (num_stages, rows, cols, channels)
    '''
    fig=plt.figure(figsize=(15, 15))
    columns = 10
    rows = 1
    for i in range(columns*rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(stages[i].astype('uint8'))
        plt.axis('off')
    plt.show()


def deletion_scores(model, norm_array_2d, image_tensor, num_stages, batch_size, show_stages):

    scores = []
    # get score of original image
    arr_3d = helpers.load_image_to_3d_array(image_tensor, target_size)
    input_arr_4d = np.expand_dims(arr_3d, axis=0)
    pred = get_preds_array(model, input_arr_4d)    
    scores.extend(pred)
    # go through batches of stages
    importances = norm_array_2d.copy()
    for _ in range(np.math.ceil(num_stages//batch_size)):
        # compute stages batch
        stages = get_deletion_stages_batch(arr_3d, importances, num_stages, batch_size)
        # get scores for stages batch
        preds = get_preds_array(model, stages)    
        scores.extend(preds)
        if show_stages == True:
            visualize_stages(stages)
    # get score of black image
    arr_3d = np.zeros( shape=target_size+(3,), dtype='float32')
    input_arr_4d = np.expand_dims(arr_3d, axis=0)
    pred = get_preds_array(model, input_arr_4d)    
    scores.extend(pred)
    
    return scores


# def get_insertion_stages_batch(original, arr_3d, importances, num_stages, batch_size):
#     '''
#     Get batch_size stages of insertion for current importances.

#     Parameters
#         original: array with 3 dimensions (rows, cols, channels)    
#         arr_3d: array with 3 dimensions (rows, cols, channels)
#         importances: array with 2 dimensions (rows, cols)        
#         num_stages: int        
#         batch_size: int

#     Returns
#         stages: array with 4 dimensions (batch_size, rows, cols, channels)
#     '''
#     stages = 127 * np.ones( shape=(batch_size,)+arr_3d.shape, dtype='float32')
#     for stage in range(batch_size):
#         for _ in range(50000//num_stages):
#             ind = np.unravel_index(np.argmax(importances, axis=None), importances.shape)
#             importances[ind] = 0
#             arr_3d[ind] = original[ind].copy()
#         stages[stage] = arr_3d.copy()
    
#     return stages


# def insertion_scores(model, norm_array_2d, image_tensor, target_size, num_stages, batch_size, show_stages):
#     '''
#     Get scores for each stage of image insertion.

#     Parameters
#         model: trained model
#         norm_array_2d: array with 2 dimensions (rows, cols)
#         image_tensor: tensor of shape (batch_size=1, image_height, image_width, channels=3)
#         target_size: tuple of (int, int)
#         num_stages: int
#         batch_size: int
#         show_stages: boolean

#     Returns
#         scores: list of float with length of 1 + batch_size * (num_stages//batch_size)
#     '''
#     scores = []
#     original = helpers.load_image_to_3d_array(image_tensor, target_size)
#     # get score of blur image
#     arr_3d = 127 * np.ones( shape=target_size+(3,), dtype='float32')
#     input_arr_4d = np.expand_dims(arr_3d, axis=0)
#     pred = get_preds_array(model, input_arr_4d)    
#     scores.extend(pred)
#     # go through batches of stages
#     importances = norm_array_2d.copy()
#     for _ in range(np.math.ceil(num_stages//batch_size)):
#         # compute stages batch
#         stages = get_insertion_stages_batch(original, arr_3d, importances, num_stages, batch_size)
#         # get scores for stages batch
#         preds = get_preds_array(model, stages)    
#         scores.extend(preds)
#         if show_stages == True:
#             visualize_stages(stages)
#     # get score of original image
#     input_arr_4d = np.expand_dims(original, axis=0)
#     pred = get_preds_array(model, input_arr_4d)    
#     scores.extend(pred)
    
#     return scores


def visualize_scores(scores, is_insert: bool):
    '''
    Visualize scores.

    Parameters
        scores: list of float
        type: boolean
    '''
    plt.plot(scores)
    title = 'εισαγμένων' if is_insert else 'διαγραμμένων'
    plt.xlabel('ποσοστό ' + title + ' εικονοστοιχείων')
    plt.xlim([0,100])
    plt.ylabel('πιθανότητα πρόβλεψης')
    plt.ylim([0,1])
    plt.show()


def area_under_curve(scores):
    '''
    Calculate area under curve of scores using trapezoid rule.

    Parameters
        scores: list of float

    Returns
        auc: float
    '''
    auc = 0
    for idx in range(len(scores)-1):
        auc += (scores[idx]+scores[idx])/2
    auc /= (len(scores)-1)
    return auc
