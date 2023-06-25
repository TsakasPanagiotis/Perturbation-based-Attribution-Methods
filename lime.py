import skimage.io
import skimage.segmentation
import numpy as np
import torch
import torch.nn.functional as F
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

def perturb_image(img, perturbation, segments):
    '''
    Params
        img: numpy.ndarray (224, 224, 3)
        perturbation: numpy.ndarray (num_superpixels,)
        segments: numpy.ndarray (224, 224)
    Return
        ...: numpy.ndarray (224, 224, 3)
    '''
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    mask[np.isin(segments, active_pixels)] = 1

    return img * np.expand_dims(mask, axis=-1)


def get_lime(image_tensor:torch.Tensor, model):

    img = np.transpose(image_tensor[0].detach().cpu().numpy(), (1,2,0))

    superpixels = skimage.segmentation.quickshift(img, kernel_size=2, max_dist=100, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]

    print('num_superpixels', num_superpixels, np.unique(superpixels, return_counts=True))

    num_perturb = 150
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(img, pert,superpixels).transpose((2,0,1))[np.newaxis]
        input = torch.tensor(perturbed_img, dtype=torch.float32, device=image_tensor.device)
        output = model(input)
        output = F.softmax(output, dim=1)
        predictions.append(output.detach().cpu().numpy())
    predictions = np.array(predictions)

    original_image = np.ones(num_superpixels)[np.newaxis,:]
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()

    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2))

    simpler_model = LinearRegression()
    simpler_model.fit(perturbations, predictions.squeeze(), sample_weight=weights)
    coeff = simpler_model.coef_[0]

    num_top_features = 2
    top_features = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features] = True

    norm_array_2d = perturb_image(np.ones_like(img) * 255, mask, superpixels).astype(np.uint8)[:,:,0]

    return norm_array_2d
