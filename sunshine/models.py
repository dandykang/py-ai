import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers, activations, Input

# 此处只是针对序列模型操作进行层数封装
# 并没有自定义实现layers和model
# 自定义层和模型 参考 https://www.tensorflow.org/guide/keras/custom_layers_and_models
class ModelSequential():
    def __init__(self, input_shape,  dense_info):
        super(ModelSequential, self).__init__()
        self.dense_list = []
        i = 0
        for cell_cnt, activ_func in dense_info:
            if i ==0 :
                self.dense_list.append(layers.Dense(cell_cnt,
                                                    input_shape=input_shape,
                                                    activation=activ_func))
            else:
                self.dense_list.append(layers.Dense(cell_cnt, activation=activ_func))
            i = i + 1

    def build(self):
        model = tf.keras.Sequential()
        dense_len = len(self.dense_list)
        i = 0
        while i < dense_len:
            model.add(self.dense_list[i])
            i = i + 1

        return model



if __name__ == '__main__':
    input_shape = (5,)
    dense_info = [(6, 'relu'), (6, 'relu'), (1, activations.sigmoid)]
    model = ModelSequential(input_shape, dense_info).build()
    model.summary()
