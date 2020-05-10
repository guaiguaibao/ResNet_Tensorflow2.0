from tensorflow.keras import layers, Model, Sequential


# 构建resnet18和resnet34使用的残差模块
class BasicBlock(layers.Layer):
    expansion = 1
    # 在这类残差模块中，前后两个3*3卷积层的kernel个数是相同的
    # expansion是定义的一个类属性，通过他可以对BasicBlock和Bottleneck进行区分，从而搭建不同的层的网络

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        # 参数中的out_channel是指模块中两个3*3卷积层的kernel的个数
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        # 使用了BN层，所以卷积层的use_bias设置为False
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # momentum参数默认是0.99
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        # 这里的downsample指skip connection上对特征图进行下采样的操作
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        # 如果需要对特征图进行下采样，则skip connection上也要对输入进行下采样操作
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


# 构建resnet50 resnet101和resnet152使用的bottleneck的残差模块
class Bottleneck(layers.Layer):
    expansion = 4
    # 在带有bottleneck的残差模块中，最后1*1卷积层的kernel个数是前两个卷积层(1*1和3*3)的kernel个数的4倍

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        # 这类的out_channel参数指的是残差模块前两层的kernel个数，最后一层的kernel个数是其4倍
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        # 这里给每一层取得名字都是为了后面迁移学习预训练的权重相对应的
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x


# _make_layer函数是搭建多个残差模块构成更大的模块
def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    # 参数中的channel是指残差模块中第一层卷积kernel的个数
    downsample = None
    # 这里downsample不仅仅是为了下采样同时调整通道数，还有只调整通道数的作用
    if strides != 1 or in_channel != channel * block.expansion:
        # 通过strides来判断是否是下采样模块，通过输入通道数和输出通道数来判断是否是只需调整通道数的模块(resnet50 101 152的conv2模块)
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


# 搭建整个网络，这里是用functional API，subclassing API在subclassed_model.py文件中
def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # 参数中的block指的是哪一种残差模块，是basicblock还是bottleneck。blocks_num是一个列表，指定了每一个块中残差模块的个数，include_top指的是最后的GlobalAveragePooling和全连接层

    # tensorflow中的tensor通道排序是NHWC
    # (None, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    # 使用_make_layer来搭建conv2 conv3 conv4 conv5四个模块的残差模块的堆叠
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    # conv2中的卷积层不需要进行下采样，所以strides默认为1，64指的是第一个卷积层的kernel个数
    # 对于resnet50 101 152，conv2模块虽然不需要进行下采样，但是通道数要进行调整
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        # GlobalAvgPool2D有两个功能，pooling和flatten，结果是一个一维向量
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x
        # 如果include_top是False，那么直接将卷积之后的特征图输出，后面可以做其他的操作

    model = Model(inputs=input_image, outputs=predict)

    return model


def resnet34(im_width=224, im_height=224, num_classes=1000):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes)


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top)

