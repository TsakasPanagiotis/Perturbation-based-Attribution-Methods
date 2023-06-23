import os

import tensorflow_datasets as tfds


if __name__ == '__main__':

    data_dir = './imagenet_v2'

    # download imagenet_v2 dataset in data_dir
    # takes about 4 GigaBytes of space
    ds = tfds.load(
        name='imagenet_v2',
        split='test',
        data_dir=data_dir,
        batch_size=1,
        shuffle_files=False,
        download=True
    )

    # path of extracted imagenet_v2 dataset
    directory = data_dir + '/downloads/extracted'

    paths = []

    # gather the paths of all extracted images
    for (dirpath, dirnames, filenames) in os.walk(directory):
        paths += [os.path.join(dirpath, file) for file in filenames]

    # write paths to all file
    with open('all.txt', 'w') as all_file:
        for path in paths:
            all_file.write(path + '\n')

    # create empty done file
    with open('done.txt', 'w'): pass

    # create empty folder for results
    newpath = './results_imagenet'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
