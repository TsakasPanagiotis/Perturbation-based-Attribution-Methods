from tqdm import tqdm
import tensorflow as tf

import helpers
import metrics


if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # load trained VGG16
    model = helpers.load_vgg16('linear')
    target_size = (224,224)

    # read paths of images that have already been used
    done = []
    with open('done.txt', 'r') as done_file:
        for line in done_file:
            done.append(line.rstrip())


    with open('all.txt', 'r') as all_file, open('done.txt', 'a') as done_file:

        for line in tqdm(all_file, total=10_000):

            path = line.rstrip()

            # ignore paths of images that have already been used
            if path in done:
                continue

            # prepare image for model
            image_file = tf.io.read_file(path)
            image_jpeg = tf.io.decode_jpeg(image_file, channels=3)
            image_tensor = tf.expand_dims(image_jpeg, axis=0)

            metrics.perform('vanilla', image_tensor, target_size, model, './results_imagenet/vanilla_results.csv')
            metrics.perform('integrated', image_tensor, target_size, model, './results_imagenet/integrated_results.csv')
            metrics.perform('smooth_vanilla', image_tensor, target_size, model, './results_imagenet/smooth_vanilla_results.csv')
            metrics.perform('smooth_integrated', image_tensor, target_size, model, './results_imagenet/smooth_integrated_results.csv')
            metrics.perform('rise_vanilla', image_tensor, target_size, model, './results_imagenet/rise_vanilla_results.csv')
            metrics.perform('rise_integrated', image_tensor, target_size, model, './results_imagenet/rise_integrated_results.csv')

            # add path of current image to paths of used images
            done_file.write(path + '\n')
