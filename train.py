import tensorflow as tf
from model import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from dataset import Dataset
import argparse
import os

# ** Define parameter
SOURCE_PATH = '/home/dunglt/cmnd/dung/data/process_extract_0606/extract_0606_id_number'
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, type=str, help='path to image source')
parser.add_argument('--label_path', required=True, type=str, help='path to label')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--charset', default='0123456789>', type=str, help='characters set')
parser.add_argument('--max_length', default=12, type=int, help='max sequence length')
parser.add_argument('--input_h', default=64, type=int, help='input height')
parser.add_argument('--input_w', default=416, type=int, help='input width')

args = parser.parse_args()
# ** Prepare data ** #
print('---------- loading data ! ----------')
traningset = Dataset(args.data_source_path, args.label_path, batch_size=args.batch_size,
                input_size=(args.input_w, args.input_h), max_sequence_length=args.max_length)
print('Found %d images in training set'%len(traningset)*args.batch_size)
valid_source_path = SOURCE_PATH + '/valid'
valid_label_path = SOURCE_PATH + '/valid/label.txt'
validset = Dataset(valid_source_path, valid_label_path, batch_size=args.batch_size,
                input_size=(args.input_w, args.input_h), max_sequence_length=args.max_length)
print('Found %d images in validation set'%len(validset)*args.batch_size)

# ** Get model ** #
print('------------ loading model ! ----------')
ckpt = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=True)
tensorboard = TensorBoard()
if not os.path.exists('best_model.h5'):
    model = Model(input_shape=(args.input_h, args.input_w, 3), max_sequence_length=args.max_length,
                charset_length=len(args.charset), feature_extractor='Efficient',
                sequence_modeling='Bilstm').make_model()
    with open('model_arc.json', 'w') as f:
        f.write(model.to_json())
else:
    with open('model_arc.json', 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights('best_model.h5')

optimizer = Adam()
model.compile(loss=binary_crossentropy, optimizer=optimizer)
print('---------- start training ! ------------')
model.fit_generator(traningset, steps_per_epoch=len(traningset), epochs=100, callbacks=[ckpt, tensorboard], validation_data=validset,
                    validation_steps=len(validset), workers=4)

print('---------- Done ! ---------- ')