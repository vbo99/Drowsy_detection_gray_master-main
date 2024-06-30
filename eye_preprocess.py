
import numpy as np
import os
from six.moves import cPickle as pickle
import cv2

openDirs = ['dataset/OpenFace']
closeDirs = ['dataset/ClosedFace']

def generate_dataset(type, dirData):
    dataset = []
    i = 0
    for dir in dirData:
        for filename in os.listdir(dir):
            if filename.endswith('.jpg') or filename.endswith('.bmp'):
                im = cv2.imread(dir + '/' + filename)
                # Convert to grayscale image
                #im[:, :, 0] = 0
                #im[:, :, 1] = 0
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = np.expand_dims(im, axis=2) 
                dataset.append(im)
                i += 1

    if type == 1:
        labels = np.ones([len(dataset), 1], dtype=int)
    else:
        labels = np.zeros([len(dataset), 1], dtype=int)
    return dataset, labels

def save_train_and_test_set(dataset, labels, ratio, pickle_file):
    split = int(len(dataset) * ratio)
    train_dataset = dataset[:split]
    train_labels = labels[:split]
    test_dataset = dataset[split:]
    test_labels = labels[split:]

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

# Main
if __name__ == '__main__':
    dataset_open, labels_open = generate_dataset(1, openDirs)
    dataset_closed, labels_closed = generate_dataset(0, closeDirs)

    ratio = 0.9

    pickle_file_open = 'open_eyes_all.pickle'
    pickle_file_closed = 'closed_eyes_all.pickle'

    # Save open dataset to pickle file
    save_train_and_test_set(dataset_open, labels_open, ratio, pickle_file_open)
    # Save close dataset to pickle file
    save_train_and_test_set(dataset_closed, labels_closed, ratio, pickle_file_closed)






