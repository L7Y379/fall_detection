import json

from app import train_stop
from function.parameter import *
from keras.layers import LSTM,Input, Dense,Flatten,MaxPooling2D,TimeDistributed,Bidirectional, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from function.service import *
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import math
import os
import time
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 下面为我们的模型网络设置
nb_time_steps = 200  #时间序列长度
nb_input_vector = 90 #输入序列
ww=1
img_rows = 15
img_cols = 18
channels = 1
img_shape = (200,img_rows, img_cols, channels)
epochs = 60
batch_size = 100
latent_dim = 90
def send_msg(conn, msg_bytes):
    """
    WebSocket服务端向客户端发送消息
    :param conn: 客户端连接到服务器端的socket对象,即： conn,address = socket.accept()
    :param msg_bytes: 向客户端发送的字节
    :return:
    """
    import struct

    token = b"\x81"  # 接收的第一字节，一般都是x81不变
    length = len(msg_bytes)
    if length < 126:
        token += struct.pack("B", length)
    elif length <= 0xFFFF:
        token += struct.pack("!BH", 126, length)
    else:
        token += struct.pack("!BQ", 127, length)

    msg = token + msg_bytes
    conn.send(msg)
    return True
def build_cnn(img_shape):
    cnn = Sequential()
    cnn.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same',input_shape=img_shape)))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3),activation='relu',strides=(1,1), padding='same')))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same')))
    cnn.add(TimeDistributed(Flatten()))
    cnn.add(TimeDistributed(Dense(90, activation="relu")))
    img = Input(shape=img_shape)
    latent_repr = cnn(img)
    return Model(img, latent_repr)
def build_rnn():
    rnn=Sequential()
    rnn.add(Bidirectional(LSTM(units=120, input_shape=(nb_time_steps, nb_input_vector))))
    rnn.add(Dense(500, activation="relu"))
    rnn.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = rnn(encoded_repr)
    return Model(encoded_repr, validity)
def build_dis():
    dis = Sequential()
    dis.add(Bidirectional(LSTM(units=120, input_shape=(nb_time_steps, nb_input_vector))))
    dis.add(Dense(500, activation="relu"))
    dis.add(Dense(9, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = dis(encoded_repr)
    return Model(encoded_repr, validity)

def train(train_feature,train_label,model_path,len_dir,dirName,model_name):
    print("train_feature" + str(train_feature.shape))
    print("train_label" + str(train_label.shape))

    #全局归化为0~1
    a=train_feature
    train_feature = (train_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))

    # # 列归化为0~1
    # min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    # all = np.concatenate((train_feature, test_feature), axis=0)
    # all = np.concatenate((all, train_feature_ot), axis=0)
    # all = min_max_scaler.fit_transform(all)
    # train_feature = all[:len(train_feature)]
    # test_feature = all[len(train_feature):(len(train_feature) + len(test_feature))]
    # train_feature_ot = all[(len(train_feature) + len(test_feature)):]

    # 行归化为0~1
    # min_max_scaler = MinMaxScaler(feature_range=[0,1])
    # all = np.concatenate((train_feature, test_feature), axis=0)
    # all = np.concatenate((all, train_feature_ot), axis=0)
    # all=all.T
    # all= min_max_scaler.fit_transform(all)
    # all=all.T
    # train_feature = all[:len(train_feature)]
    # test_feature = all[len(train_feature):(len(train_feature)+len(test_feature))]
    # train_feature_ot = all[(len(train_feature)+len(test_feature)):]

    train_feature = train_feature.reshape([int(train_feature.shape[0] / 200), 200, img_rows, img_cols])
    train_feature = np.expand_dims(train_feature, axis=4)

    opt = Adam(0.0002, 0.5)
    cnn = build_cnn(img_shape)
    rnn = build_rnn()
    rnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    img3 = Input(shape=img_shape)
    encoded_repr3 = cnn(img3)
    validity1 = rnn(encoded_repr3)
    crnn_model = Model(img3, validity1)
    crnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # crnn_model.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000crnn_model.h5')
    # dis.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000dis.h5')
    k = 0
    acc=0
    for epoch in range(epochs):

        idx = np.random.randint(0, train_feature.shape[0], batch_size)
        imgs = train_feature[idx]
        crnn_loss = crnn_model.train_on_batch(imgs, train_label[idx])
        if epoch % 10 == 0:
            print("%d [fall_detection_loss: %f,acc: %.2f%%]" % (epoch, crnn_loss[0], 100 * crnn_loss[1]))
            acc_on=format(100 * crnn_loss[1], '.1f')
            data = {"code": 2, "train_down": 0, "len_dir": len_dir,"epoch": epoch,"epochs": epochs,"acc": acc_on,"model_name":model_name,"dirName":dirName}
            data_json = json.dumps(data)
            i = 0
            tem = len(conns_pool)
            print("conns_pool", len(conns_pool))
            while i < tem:
                try:
                    send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
                except BrokenPipeError:
                    conns_pool.pop(i)
                    tem = len(conns_pool)
                except ConnectionAbortedError:
                    conns_pool.pop(i)
                    tem = len(conns_pool)
                i = i + 1
        if (epoch!=0 and epoch % 50==0 and acc < crnn_loss[1]):
            acc = crnn_loss[1]
            # if os.path.exists(model_path):
            #     os.remove(model_path)
            crnn_model.save_weights(model_path)

    K.clear_session()
    filename = "data_model_dir//map//states.json"
    with open(filename, 'r') as f:
        line = f.readline()
        states = json.loads(line)
    states['modelTrain'] = 1  # 设置状态为训练完成
    with open(filename, 'w+') as f:
        json.dump(states, f)
    time.sleep(1)
    if not os.path.exists(model_path):
        model_on=0
    else:
        model_on=1
    acc_on = format(100 * crnn_loss[1], '.1f')
    data = {"code": 2, "train_down": 1, "len_dir": len_dir, "epoch": epochs, "epochs": epochs,
            "acc": acc_on,"model_on":model_on,"model_name":model_name,"dirName":dirName }
    data_json = json.dumps(data)
    i = 0
    tem = len(conns_pool)
    print("conns_pool", len(conns_pool))
    while i < tem:
        try:
            send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
        except BrokenPipeError:
            conns_pool.pop(i)
            tem = len(conns_pool)
        except ConnectionAbortedError:
            conns_pool.pop(i)
            tem = len(conns_pool)
        i = i + 1
    time.sleep(1)
    train_thread[0] = 0
    print("训练完成")
    #train_stop()

def test(test_feature,modelName):
    print("test_feature" + str(test_feature.shape))

    #全局归化为0~1
    a=test_feature
    test_feature = (test_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
    test_feature = test_feature.reshape([int(test_feature.shape[0] / 200), 200, img_rows, img_cols])
    test_feature = np.expand_dims(test_feature, axis=4)

    #opt = Adam(0.0002, 0.5)
    cnn = build_cnn(img_shape)
    rnn = build_rnn()
    #rnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    img3 = Input(shape=img_shape)
    encoded_repr3 = cnn(img3)
    validity1 = rnn(encoded_repr3)
    crnn_model = Model(img3, validity1)
    #crnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    crnn_model.load_weights(modelName)
    label = crnn_model.predict(test_feature)#(1,2)
    K.clear_session()
    print("label", label)
    if(label[:,0:1]>=label[:,1:2]):
        result=0
    else:
        result=1
    return result

def test_on(test_feature,modelName):
    print("test_feature" + str(test_feature.shape))

    #全局归化为0~1
    a=test_feature
    test_feature = (test_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
    #test_feature=np.tile(test_feature, (1, 270))
    test_feature = test_feature.reshape([int(test_feature.shape[0] / 200), 200, img_rows, img_cols])
    test_feature = np.expand_dims(test_feature, axis=4)

    #opt = Adam(0.0002, 0.5)
    cnn = build_cnn(img_shape)
    rnn = build_rnn()
    #rnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    img3 = Input(shape=img_shape)
    encoded_repr3 = cnn(img3)
    validity1 = rnn(encoded_repr3)
    crnn_model = Model(img3, validity1)
    #crnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    crnn_model.load_weights(modelName)
    label = crnn_model.predict(test_feature)#(1,2)
    K.clear_session()
    print("label", label)
    if(label[:,0:1]>=label[:,1:2]):
        result=0
    else:
        result=1
    return result

