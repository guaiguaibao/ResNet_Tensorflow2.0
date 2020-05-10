from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import resnet50
import tensorflow as tf
import json
import os
import PIL.Image as im
import numpy as np

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
train_dir = image_path + "train"
validation_dir = image_path + "val"

im_height = 224
im_width = 224
batch_size = 16
epochs = 20

# 这三个参数是ImageNet数据集中图像的三个通道的平均，因为下面要基于迁移学习来训练网络，预训练的参数对应了图像通道减去均值的操作
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


# 图像通道减去ImageNet中的通道均值
def pre_function(img):
    # img = im.open('test.jpg')
    # img = np.array(img).astype(np.float32)
    img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

    return img


# data generator with data augmentation
train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                           preprocessing_function=pre_function)

validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
# img, _ = next(train_data_gen)
total_val = val_data_gen.n

# 这里之所以用resnet50，因为tensorflow官方只给出了resnet50和resnet101的预训练参数
feature = resnet50(num_classes=5, include_top=False)
# feature.build((None, 224, 224, 3))  # when using subclass model，传入batch_input_shape参数
feature.load_weights('pretrain_weights.ckpt')
feature.trainable = False
feature.summary()
# resnet50参数量大约是2000万，resnet152的参数有644MB

model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(5),
                             tf.keras.layers.Softmax()])
# model.build((None, 224, 224, 3))
model.summary()

# using keras low level api for training
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        # 训练过程中training参数设置为True，使BN层进行学习
        loss = loss_object(labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)


@tf.function
def test_step(images, labels):
    output = model(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)


best_test_loss = float('inf')
for epoch in range(1, epochs + 1):
    train_loss.reset_states()  # clear history info
    train_accuracy.reset_states()  # clear history info
    test_loss.reset_states()  # clear history info
    test_accuracy.reset_states()  # clear history info

    # train
    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_step(images, labels)

        # print train process
        rate = (step + 1) / (total_train // batch_size)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        acc = train_accuracy.result().numpy()
        print("\r[{}]train acc: {:^3.0f}%[{}->{}]{:.4f}".format(epoch, int(rate * 100), a, b, acc), end="")
        # 该进度条输出的例子：[1]train acc: 35 %[************->................]0.6005
        # {:^3.0f}的解释：^代表居中，3表示长度，format填充不足3自动填充空格，.0f表示小数点精确度到个位
    print()

    # validate
    for step in range(total_val // batch_size):
        test_images, test_labels = next(val_data_gen)
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    if test_loss.result() < best_test_loss:
        best_test_loss = test_loss.result()
        model.save_weights("./save_weights/resNet_50.ckpt", save_format="tf")
# 如果模型中不加softmax层的话，最终的验证集准确率会更高一些，这就是之前说的数值稳定性
