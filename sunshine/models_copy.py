import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers, activations, Input


class ModelSequential(Model):
    def __init__(self, dense_info):
        super(ModelSequential, self).__init__()
        self.dense_list = []
        for cell_cnt, activ_func in dense_info:
            self.dense_list.append(layers.Dense(cell_cnt, activation=activ_func))

        # self.f1 = layers.Dense(6,activation='relu')
        # self.f2 = layers.Dense(4,activation='relu')
        # self.f3 = layers.Dense(1,activation='sigmoid')


    def call(self, x):
        dense_len = len(self.dense_list)
        i = 0
        while i < dense_len:
            x = self.dense_list[i](x)
            i = i + 1

        return x


def normalSequential():
    # 建立序列模型
    model = tf.keras.Sequential()
    # 添加隐藏层,神经元为6个，输入类型为一维数组的5个特征
    model.add(tf.keras.layers.Dense(6, input_shape=(5,), activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
    return model



if __name__ == '__main__':
    dense_info = [(6, 'relu'), (6, 'relu'), (1, activations.sigmoid)]
    model = ModelSequential(dense_info)
    # model.build(input_shape=(5,))
    model.build(input_shape=())
    model.summary()
