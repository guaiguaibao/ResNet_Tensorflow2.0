from model import resnet50
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf

im_height = 224
im_width = 224

# load image
img = Image.open("../tulip.jpg")
# resize image to 224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# scaling pixel value to (0-1)，准确的说这里不是为了归一化，仅仅是为了zero-mean的操作
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
img = np.array(img).astype(np.float32)
img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

feature = resnet50(num_classes=5, include_top=False)
feature.trainable = False
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(5),
                             tf.keras.layers.Softmax()])
# model.build((None, 224, 224, 3))  # when using subclass model
model.load_weights('./save_weights/resNet_50.ckpt')
result = model.predict(img)
prediction = np.squeeze(result)
predict_class = np.argmax(result)
print(class_indict[str(predict_class)], prediction[predict_class])
plt.show()
