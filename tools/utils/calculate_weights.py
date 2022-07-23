import os
from tqdm import tqdm
import numpy as np


def calculate_weigths_labels(data_path, dataset, dataloader, num_class):
    # Create an instance from the dataloaders loader
    z = np.zeros((num_class,))
    # Initialize tqdm
    b_bar = tqdm(dataloader, desc='Calculating classes weights')
    for sample in b_bar:
        y = sample['target']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_class)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_class)
        z += count_l
    b_bar.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(data_path, dataset + '_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret
