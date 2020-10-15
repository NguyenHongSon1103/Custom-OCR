import tensorflow as tf
import numpy as np
import cv2
import os

charset = '0123456789>'
def chars2onehot(chars=9, max_sequence_length=12):
    onehot_tensor = np.zeros((max_sequence_length, len(charset))) #create matrix (12, 11)
    for i, char in enumerate(chars):
        onehot_tensor[i][int(char)] = 1
    for i in range(len(chars), max_sequence_length):
        onehot_tensor[i][-1] = 1
    return onehot_tensor


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, data_source_path, label_path, batch_size=32, input_size=(416, 46), max_sequence_length=15):
        with open(label_path, 'r') as f:
            self.labels = f.read().split('\n')
        self.batch_size = batch_size
        self.data_source_path = data_source_path
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
    
    def on_epoch_end(self):
        np.random.shuffle(self.labels)
    
    def __len__(self):
        return len(self.labels) // self.batch_size
    
    def __get_item__(self, idx):
        raws = self.labels[self.batch_size*idx:self.batch_size*(idx+1)]
        raws = [label.split('   ') for label in raws]
        filenames = [label[0] for label in raws]
        labels = [label[1][:-1] for label in raws]
        images = []
        # ** Prepare for images ** #
        for filename in filenames:
            fp = os.path.join(self.data_source_path, filename)
            img = cv2.imread(fp)
            img = cv2.resize(img, self.input_size) / 255.0
            images.append(img)
        
        # ** Prepare for labels ** #
        onehot_labels = []
        for label in labels:
            onehot = chars2onehot(label, max_sequence_length=self.max_sequence_length)
            onehot_labels.append(onehot)
        onehot_labels = np.array(onehot_labels)
        images = np.array(images)
        return (images, onehot_labels)
            
        


