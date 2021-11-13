import  csv
import threading
from itertools import chain
import  json
import pandas as pd
import numpy as np
from function.Read_Bf import read_bf_file
from function.pre_deal_filter import *
import os


# names = {'modelTrain':0, 'localization':1}
# filename = 'data_model_dir//map//states.json'
# #
# with open(filename,'w+') as f:
#     json.dump(names,f)
#
# with open(filename,'r') as f:
#     line = f.readline()
#     print(line)
#     states = json.loads(line)
#     print(states['modelTrain'])
# #
# print(states['localization'])

# def fun_timer():
#     print("hello")
#     global timer
#     timer = threading.Timer(3,fun_timer)
#     timer.start()
#
# timer = threading.Timer(3,fun_timer)
# timer.start()

frompath = r'data_model_dir\data_dir\test_tk_path_1027\two_open_100_2.dat'
csvpath =r'data_model_dir\data_dir\test_tk_path_1027\smooth_data\two_open_100_2.csv'

curpin = 0
stream, curpin = read_bf_file(frompath, curpin)  ##stream里包含的是幅值了么  curpin是啥
amplitude_list = []
for i in range(len(stream)):
    data = stream[i]["csi"]
    if data.shape == (3, 3, 30):
        amplitude_list.append(abs(data))
print("amplitude_list:", len(amplitude_list))
amplitude_list = np.array(amplitude_list)
stream_len = len(amplitude_list)
amplitude_list_new = amplitude_list.reshape((stream_len, -1))

data_flag = 0
for i in range(0, amplitude_list_new.shape[1]):
    # print(i)
    data1 = amplitude_list_new[:, i].reshape(-1)
    datatmp = smooth(hampel(data1), 25)

    if data_flag == 0:
        data = datatmp.reshape(-1, 1)
        data_flag = 1
    else:
        data = np.column_stack((data, datatmp.reshape(-1, 1)))


test=pd.DataFrame(data=data)
test.to_csv(csvpath,columns=None,index=None,header=None,encoding="gbk")
# print(data.shape)
# with open(csvpath, 'a+', newline='') as f:
#     csv_write = csv.writer(f)
#     csv_write.writerow(data)

print("finished")