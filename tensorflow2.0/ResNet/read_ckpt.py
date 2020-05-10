import tensorflow as tf


# 该文件将官方提供的预训练权重转换为我们需要的权重
def rename_var(ckpt_path, new_ckpt_path):
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        # 将所有预训练权重放入var_list中
        var_list = tf.train.list_variables(ckpt_path)
        for var_name, shape in var_list:
            print(var_name)
            # 如果层名称不是我们想要的，直接跳过
            if var_name in except_list:
                continue
            var = tf.train.load_variable(ckpt_path, var_name)
            # 如果某个权重层是我们需要的，通过下面的操作将层的名称改为我们定义的模型中的层名称
            new_var_name = var_name.replace('resnet_v1_50/', "")
            new_var_name = new_var_name.replace("bottleneck_v1/", "")
            new_var_name = new_var_name.replace("shortcut/weights", "shortcut/conv1/kernel")
            new_var_name = new_var_name.replace("weights", "kernel")
            new_var_name = new_var_name.replace("biases", "bias")
            re_var = tf.Variable(var, name=new_var_name)
            new_var_list.append(re_var)

        # 重新定义最后一层全连接层，对于resnet50，最后卷积层的输出是2048，然后全连接层需要5个神经元
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([2048, 5]), name="logits/kernel")
        # re_var = tf.Variable(tf.random.truncated_normal([2048, 5]), name="logits/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([5]), name="logits/bias")
        # re_var = tf.Variable(tf.random.truncated_normal([5]), name="logits/bias")
        new_var_list.append(re_var)
        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


# 建立一个列表，其中包含了我们要筛除的层结构，最后两个是全连接层的偏置和权重
except_list = ['global_step', 'resnet_v1_50/mean_rgb', 'resnet_v1_50/logits/biases', 'resnet_v1_50/logits/weights']
ckpt_path = './resnet_v1_50.ckpt'
new_ckpt_path = './pretrain_weights.ckpt'
new_var_list = []
rename_var(ckpt_path, new_ckpt_path)
"""
该文件可以将tensorflow官方提供的权重文件resnet_v1_50.ckpt中的权重进行转换，主要是为了删除某些层和将权重层的名称改为我们自己定义的模型的层名称。最后输出是两个文件：pretrain_weights.ckpt.index和pretrain_weights.ckpt.data-00000。然后在train.py脚本中加载的预训练参数文件就是这两个文件，从而进行迁移学习。train.py训练之后保存最终的模型权重 - resNet_50.ckpt。然后在predict.py做预测的时候，就直接加载resNet_50.ckpt中的最终权重。
"""
"""
该文件是针对tensorflow github仓库中slim文件夹中的预训练参数，实现方法不是很高级
另外之所以在训练的时候对图像的通道减去了ImageNet中的通道均值，是因为在slim文件夹下的preprocessing文件夹中有个preprocessing_factory.py文件，该文件中的preprocessing_fn_map字典提供了不同预训练模型对应的图像预处理函数。resnet_v1_50对应的是preprocessing文件夹中的vgg_preprocessing文件。查看vgg_preprocessing文件之后，发现了_mean_image_subtraction函数，他就是对图像进行预处理的函数，其中就有ImageNet中的三个通道的均值。

所以，如果要使用其他模型的预训练权重，只需要按照上面的步骤，找到对应的图像预处理方法，然后在自己的训练脚本中与之保持一致，就可以使用tensorflow提供的权重进行迁移学习了。
"""
