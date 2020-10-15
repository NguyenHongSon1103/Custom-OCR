import tensorflow.keras.models as models
import os
import cv2
import numpy as np
from time import time
from sys import argv

charset = '0123456789>'
model_path = "ID_Recognition/model_arc.json"
weights_path = "ID_Recognition/best_model.h5"
with open(model_path, 'r') as f:
    model = models.model_from_json(f.read())
model.load_weights(weights_path)

if len(argv) < 2:
    path = r'E:\VNG_Intern\FakeID_Detect\Dataset\CMT\test\valid'
else:
    path = argv[1]
for name in os.listdir(path):
    fp = os.path.join(path, name)
    img = cv2.imread(fp)
    img = cv2.resize(img, (416, 64))
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ = img / 255.0
    tensor = np.expand_dims(img_, 0)
    s = time()
    prediction = model.predict(tensor)[0]
    prediction = np.argmax(prediction, 1)
    print(prediction)
    result = ''
    for pred in prediction:
        char = charset[pred]
        # if char == '>':
        #     break
        result += char
    print(result, time() - s)
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit(-1)