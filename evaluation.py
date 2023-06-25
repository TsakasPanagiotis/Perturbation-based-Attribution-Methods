import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import time


def get_deletion_stages_batch(arr_3d, importances, num_stages, batch_size):

    arr_3d = np.transpose(arr_3d, (1,2,0))    

    stages = np.zeros( shape=(batch_size,)+arr_3d.shape, dtype='float32')
    for stage in range(batch_size):
        for _ in range(900//num_stages):
            ind = np.unravel_index(np.argmax(importances, axis=None), importances.shape)
            importances[ind] = 0
            arr_3d[ind] = np.array([0,0,0], dtype='float32')
        stages[stage] = arr_3d.copy()
    
    stages = np.transpose(stages, (0,3,1,2))
    
    return stages


def visualize_stages(stages):

    stages = np.transpose(stages, (0,2,3,1))

    fig=plt.figure(figsize=(15, 15))
    columns = 10
    rows = 1
    for i in range(columns*rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(stages[i])
        plt.axis('off')
    plt.savefig(f'stages_{time.time()}.png')
    plt.close()


def deletion_scores(model, norm_array_2d, image_tensor, target, num_stages, batch_size, show_stages):

    print('image_tensor', image_tensor.shape, image_tensor.dtype)

    scores = []
    # get score of original image
    logits = model(image_tensor)
    target_logits = logits[:,target]
    pred = torch.exp(target_logits) / torch.sum(torch.exp(logits), dim=1)    
    scores.extend(pred.tolist())
    # go through batches of stages
    importances = deepcopy(norm_array_2d)
    arr_3d = image_tensor[0].detach().cpu().numpy()
    for _ in range(np.math.ceil(num_stages//batch_size)):
        # compute stages batch
        stages = get_deletion_stages_batch(arr_3d, importances, num_stages, batch_size)
        # get scores for stages batch
        logits = model(torch.from_numpy(stages).to(image_tensor.device))
        target_logits = logits[:,target]
        preds = torch.exp(target_logits) / torch.sum(torch.exp(logits), dim=1)    
        scores.extend(preds.tolist())
        if show_stages == True:
            visualize_stages(stages)
    # get score of black image
    black_image_tensor = torch.zeros_like(image_tensor)
    logits = model(black_image_tensor)
    target_logits = logits[:,target]
    pred = torch.exp(target_logits) / torch.sum(torch.exp(logits), dim=1)     
    scores.extend(pred.tolist())
    
    return scores


def visualize_scores(scores, is_insert: bool):

    plt.plot(scores)
    title = 'inserted' if is_insert else 'deleted'
    plt.xlabel('percentage of ' + title + ' pixels')
    plt.xlim([0,100])
    plt.ylabel('prediction probability')
    plt.ylim([0,1])
    # plt.show()
    plt.savefig('del_curve.png')


def area_under_curve(scores):

    auc = 0
    for idx in range(len(scores)-1):
        auc += (scores[idx]+scores[idx])/2
    auc /= (len(scores)-1)
    return auc
