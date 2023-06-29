from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def preprocess_stl(dataset, classes_to_remove):

    labels = dataset.labels    
    indices_to_keep = [i for i, label in enumerate(labels) if label not in classes_to_remove]
    dataset_subset = Subset(dataset, indices_to_keep)
    labels_subset = [label for i, label in enumerate(labels) if i in indices_to_keep]
    train_indices, test_indices = train_test_split(range(len(labels_subset)), test_size=0.1, stratify=labels_subset, random_state=0)
    train_subset = Subset(dataset_subset, train_indices)
    test_subset = Subset(dataset_subset, test_indices)
    
    return train_subset, test_subset


def create_subset(args, dataset):
    subset = []
    if args.experiment == 'pneumonia':
        num_images_per_class = 50
        class_counts = [0] * 2
    elif args.experiment == 'tumor':
        num_images_per_class = 50
        class_counts = [0] * 2
    elif args.experiment == 'stl':
        num_images_per_class = 25
        class_counts = [0] * 10

    for image, label in dataset:
        if class_counts[label] < num_images_per_class:
            subset.append((image, label))
            class_counts[label] += 1
        if all(count >= num_images_per_class for count in class_counts):
            break
    return subset