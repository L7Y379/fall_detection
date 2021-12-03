import csv
import json
import os
from function.Read_Bf import read_bf_file
import pandas as pd
from function.pre_deal_filter import *
from function.network import *
import matplotlib.pyplot as plt
import ctypes
import inspect
import time



# 下面两个函数为杀死线程的函数，python没有自带的杀死线程的函数
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

def getlabel(path):
    path2=path.replace(".csv",".dat")
    col_name = ['path','label']
    filepath = 'data_model_dir\map\labels.csv'
    csv_data = pd.read_csv(filepath, names=col_name, header=None)
    label = -1
    for index, row in csv_data.iterrows():
        # print(row)
        if row["path"] == path or row["path"] == path2:
            # print(111)
            label = row["label"]
    # print("labels:",labels)
    print("label", label)
    return label

#直接取预处理后的数据训练训练
def model_train(dirname, model_path,dirName,model_name):
    k = 0 #标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    len_dir=len(dataList)
    data = {"code": 2, "train_down": 0, "len_dir": len_dir,"epoch": 0,"epochs": 0,"acc": 0,"model_name":model_name,"dirName":dirName}
    data_json = json.dumps(data)
    i = 0
    tem = len(conns_pool)
    print("conns_pool", len(conns_pool))
    while i < tem:
        try:
            send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
        except (BrokenPipeError, ConnectionAbortedError,ConnectionResetError):
            conns_pool.pop(i)
            tem = len(conns_pool)
        i = i + 1
    for i in range(0,len(dataList)):
        path = os.path.join(dirname,dataList[i])
        if os.path.isfile(path):
            temp_data=pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            if k == 0:
                raw_data=temp_data
                label = getlabel(dataList[i])
                k = 1
            else:
                raw_data = np.row_stack((raw_data, temp_data))
                label = np.row_stack((label, getlabel(dataList[i])))
    label=np_utils.to_categorical(label)
    print("raw_data:",raw_data.shape)
    print("label:", label.shape)
    t = range(10000)
    #plt.plot(t[:raw_data.shape[0]], raw_data[:, 0:1], 'r')
    #plt.show()
    #plt.savefig("D:\\my bad\\CSI_DATA\\fall_detection\\fall_detection\\data_model_dir\\data_dir\\2.png")
    data=raw_data
    label = label.astype(np.float32)
    train(data,label, model_path,len_dir,dirName,model_name)

#先预处理后训练
# def model_train(dirname, model_path):
#     k = 0 #标识符 判断数据列表是否新建
#     print("进入模型训练。。。")
#     dataList = os.listdir(dirname)
#     for i in range(0,len(dataList)):
#         path = os.path.join(dirname,dataList[i])
#         if os.path.isfile(path):
#             temp_data = get_data(path)
#             # 数据预处理
#             temp_data = pre_handle(temp_data)
#             # 数据截取
#             temp_data = data_cut(temp_data)
#             if k == 0:
#                 raw_data=temp_data
#                 label = getlabel(dataList[i])
#                 k = 1
#             else:
#                 raw_data = np.row_stack((raw_data, temp_data))
#                 label = np.row_stack((label, getlabel(dataList[i])))
#     label=np_utils.to_categorical(label)
#     print("raw_data:",raw_data.shape)
#     print("label:", label.shape)
#     t = range(10000)
#     plt.plot(t[:raw_data.shape[0]], raw_data[:, 0:1], 'r')
#     #plt.show()
#     plt.savefig("D:\\my bad\\CSI_DATA\\fall_detection\\fall_detection\\data_model_dir\\data_dir\\2.png")
#     data=raw_data
#     label = label.astype(np.float32)
#     train(data,label, model_path)


def model_test(filename,modelName):
    raw_data = get_test_data(filename)#(n,270)
    ##数据预处理
    print("执行预处理")
    data_flag = 0
    for i in range(0,raw_data.shape[1]):
        # print("raw_data.shape[1]:",raw_data.shape[1])
        data1 = raw_data[:,i].reshape(-1)
        #datatmp = smooth(hampel(data1),25)
        datatmp = smooth(data1, 25)
        if data_flag == 0:
            data = datatmp.reshape(-1,1)
            data_flag=1
        else:
            data = np.column_stack((data,datatmp.reshape(-1,1)))
    print("data_shape:",data.shape)
    data = data.astype(np.float32)
    print("执行data_cut")
    data = data_cut_threshold(data)
    if type(data) != bool:
        print(data.shape)
        result = test(data,modelName)
        return result
    else:
        return -1

def model_test_on(data,modelName):
    data = data.astype(np.float32)
    result = test_on(data,modelName)
    return result

def pre_datalist(dataList,dirPath,dirPath_pre,data_path,data_path_pre):
    data = {"code": 1, "pre_down": 0, "len_now": 0, "len_all": len(dataList), "data_path": data_path,
            "dirPath_pre": data_path_pre}
    data_json = json.dumps(data)
    i = 0
    tem = len(conns_pool)
    print("conns_pool", len(conns_pool))
    while i < tem:
        try:
            send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            conns_pool.pop(i)
            tem = len(conns_pool)
        i = i + 1
    for i in range(0, len(dataList)):
        path = os.path.join(dirPath, dataList[i])
        if os.path.isfile(path):
            temp_data = get_data(path)
            if type(temp_data) == bool:
                continue
            # 数据预处理
            temp_data = pre_handle_onlysmooth(temp_data)
            # 数据截取
            if("walk" in path):
                temp_data = data_cut_walk(temp_data)
            else:
                temp_data = data_cut(temp_data)
            print(temp_data.shape)
            path = os.path.join(dirPath_pre, dataList[i])
            path = path.replace(".dat", ".csv")
            df = pd.DataFrame(temp_data)
            # 保存 dataframe
            df.to_csv(path, encoding='utf-8', index=False, header=None)

            data = {"code": 1,"pre_down":0, "len_now":i+1,"len_all":len(dataList), "data_path":data_path,"dirPath_pre": data_path_pre}
            data_json = json.dumps(data)
            i = 0
            tem = len(conns_pool)
            print("conns_pool", len(conns_pool))
            while i < tem:
                try:
                    send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    conns_pool.pop(i)
                    tem = len(conns_pool)
                i = i + 1
            # path=path.replace(".dat",".csv")
            # os.makedirs(path)
            # with open(path, 'w',encoding='utf-8') as f:
            #     f_csv = csv.writer(f)
            #     f_csv.writerows(temp_data)
    filename = "data_model_dir//map//states.json"
    with open(filename, 'r') as f:
        line = f.readline()
        states = json.loads(line)
    states['pre_data'] = 1  # 设置状态为没有预处理
    with open(filename, 'w+') as f:
        json.dump(states, f)
    time.sleep(1)
    data = {"code": 1,"pre_down":1, "time": time.localtime(time.time()), "dirPath_pre": data_path_pre}
    data_json = json.dumps(data)
    i = 0
    tem = len(conns_pool)
    print("conns_pool", len(conns_pool))
    while i < tem:
        try:
            send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            conns_pool.pop(i)
            tem = len(conns_pool)
        i = i + 1
    predata_thread[0] = 0
def get_data(path):
    curpin = 0
    stream, curpin = read_bf_file(path, curpin) ##stream里包含的是幅值了么  curpin是啥
    amplitude_list = []
    for i in range(len(stream)):
        data = stream[i]["csi"]
        if data.shape == (3,3,30):
            amplitude_list.append(abs(data))
    print("amplitude_list:",len(amplitude_list))
    amplitude_list = np.array(amplitude_list)
    stream_len = len(amplitude_list)
    if(stream_len!=0):
        amplitude_list_new = amplitude_list.reshape((stream_len,-1))
        return amplitude_list_new
    else:
        return False
def data_cut(data):
    all_index = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        max_var = 0
        max_index = 0
        for j in range(data.shape[0]-sampleNum):
            current_var=np.var(data[j:j+sampleNum,i:i+1])
            if(current_var>=max_var):
                max_var=current_var
                max_index=j
        #print(max_var)
        all_index[i]=max_index
    #print("all_index",all_index)
    mean_index=int(np.mean(all_index))
    print("mean_index",mean_index)
    return data[mean_index:mean_index+sampleNum,:]
def data_cut_walk(data):
    #num_cut1=int((data.shape[0]-450)/2)
    if (data.shape[0] >= 421):
        num_cut1=20
        data=data[num_cut1:num_cut1+400,:]
        idx = np.array([j for j in range(0 ,400, 2)])
        data=data[idx]
    elif(data.shape[0]<421 and data.shape[0]>=221):
        num_cut1 = 20
        data = data[num_cut1:num_cut1 + sampleNum, :]
    print("walk",data.shape)
    return data
def data_cut_jz(data):

    #num_cut1=int((data.shape[0]-450)/2)
    num_cut1=20
    data=data[num_cut1:num_cut1+400,:]
    idx = np.array([j for j in range(0 ,400, 2)])
    data=data[idx]
    print("jz",data.shape)
    return data
def data_cut_threshold(data,threshold=25):
    all_index = np.zeros(data.shape[1])
    all_max_var=0
    for i in range(data.shape[1]):
        max_var = 0
        max_index = 0
        for j in range(data.shape[0]-sampleNum):
            current_var=np.var(data[j:j+sampleNum,i:i+1])
            if(current_var>=max_var):
                max_var=current_var
                max_index=j
        if(max_var>all_max_var):
            all_max_var=max_var
        all_index[i]=max_index
    if(all_max_var>threshold):
        #print("all_index",all_index)
        mean_index=int(np.mean(all_index))
        print("mean_index", mean_index)
        return data[mean_index:mean_index + sampleNum, :]
    else:
        mean_index=-1
        #print("mean_index", mean_index)
        return False
def pre_handle(raw_data):
    data_flag = 0
    for i in range(0, raw_data.shape[1]):
        #print(i)
        data1 = raw_data[:, i].reshape(-1)
        datatmp = smooth(hampel(data1), 25)

        if data_flag == 0:
            data = datatmp.reshape(-1, 1)
            data_flag = 1
        else:
            data = np.column_stack((data, datatmp.reshape(-1, 1)))
    print("data_shape:", data.shape)
    print("数据已预处理")
    return data
def pre_handle_onlysmooth(raw_data):
    data_flag = 0
    for i in range(0, raw_data.shape[1]):
        #print(i)
        data1 = raw_data[:, i].reshape(-1)
        datatmp = smooth(data1, 25)

        if data_flag == 0:
            data = datatmp.reshape(-1, 1)
            data_flag = 1
        else:
            data = np.column_stack((data, datatmp.reshape(-1, 1)))
    print("data_shape:", data.shape)
    print("数据已预处理")
    return data

def get_test_data(path):
    curpin = 0
    stream, curpin = read_bf_file(path, curpin) ##stream里包含的是幅值了么  curpin是啥
    amplitude_list = []
    for i in range(len(stream)):
        data = stream[i]["csi"]
        if data.shape == (3,3,30):
            amplitude_list.append(abs(data))
    print("amplitude_list:",len(amplitude_list))
    stream_len = len(amplitude_list)
    amplitude_list = np.array(amplitude_list)
    amplitude_list_new = amplitude_list.reshape((stream_len,-1))
    #
    return amplitude_list_new

def get_file(trans, frompath, topath,model_path):
    #time_start = time.time()
    sftp = paramiko.SFTPClient.from_transport(trans)
    curpin = 0  # 这是我们读取文件的初始位置，开始时我们设置为零
    i=0# 计数器，定期清理文件中数据
    k=0
    time.sleep(0.3)
    while True:
        try:
            sftp.get(frompath, topath)
        except IOError:
            pass
        x, curpin = read_bf_file(topath, curpin)
        print("读取文件大小" + str(len(x)))
        print("读取文件位置" + str(curpin))
        get_amplitude_phase_etc(x,model_path,k)
        i=i+1
        k=1
        if(i==15):
            ssh.exec_command("du -s " + RX_FLIE+"/test.dat")
            ssh1.exec_command("du -s " + RX_FLIE+"/test.dat")
            ssh1.exec_command(RX_FLIE+"/clean_test.sh")
            time.sleep(0.5)
            ssh1.exec_command(RX_FLIE+"/ping.sh")
            time.sleep(1)
            ssh.exec_command(RX_FLIE+"/log.sh test")
            print("i",i)
            curpin = 0
            i=0
            #print("i",i)

            # ssh.exec_command("/home/rtlab420/clean_test.sh")
            # time.sleep(1)
            # ssh.exec_command("/home/rtlab420/log.sh test")
            # curpin = 0
            # i = 0
            # print("i", i)
        time.sleep(0.5)

def get_file1(trans, frompath, topath,model_path):
    #time_start = time.time()
    sftp = paramiko.SFTPClient.from_transport(trans)
    curpin = 0  # 这是我们读取文件的初始位置，开始时我们设置为零
    i=0# 计数器，定期清理文件中数据
    time.sleep(0.3)
    k=0
    while True:
        try:
            sftp.get(frompath, topath)
        except IOError:
            pass
        x, curpin = read_bf_file(topath, curpin)
        print("读取文件大小" + str(len(x)))
        print("读取文件位置" + str(curpin))
        get_amplitude_phase_etc(x,model_path,k)
        i=i+1
        k = 1
        if(i==10):
            ssh.exec_command("du -s " + RX_FLIE+"/test.dat")
            ssh.exec_command(RX_FLIE+"/clean_test.sh")
            time.sleep(1)
            ssh.exec_command(RX_FLIE+"/log.sh test")
            print("i",i)
            curpin = 0
            i=0
        time.sleep(0.5)
amplitude_list = []
phase_list = []


# 输入是：csi数据流，我们前面使用函数获得了数据片段；应用类型诸如1表示呼吸；还有开始时间用于后面的时间模型时间训练用时计#算
#获取CSI数据流的相位和振幅并加入到我们的振幅相位列表中
def get_amplitude_phase_etc(csi_stream,model_path,k):
    len_csi_stream = len(csi_stream)
    #print("数据流的长度" + str(len_csi_stream))
    for i in range(len_csi_stream):
        data = csi_stream[i]["csi"]
        #amplitude_list.append(abs(data))
        if data.shape == (3, 3, 30):
            amplitude_list.append(abs(data))
        #phase_list.append(np.angle(data))
    #print("我们的振幅数组长度" + str(len(amplitude_list)))
    activity_realtime_test(amplitude_list,model_path,k)

#实时的活动测试（监控），调用模型进行实时的监控，使用websocket端口进行信息传递
def activity_realtime_test(amplitude_list,model_path,k):
    size = len(amplitude_list)
    #time_end = time.time()
    #print(size)
    # if size != 0:
    #     temp = np.array(amplitude_list)
    #     print(temp.shape)
    #     m = np.row_stack((distinguish_list, temp))[size:, :, :, :]
    #     print("我们的识别数组的形状：" + str(m.shape))
    # else:
    #     m = distinguish_list
    if(k==0):
        for i in range(500):
            subcarrier_amplitude_list.append(amplitude_list[0])
            subcarrier_amplitude_list.pop(0)
    for i in range(size):
        # subcarrier_amplitude_list.append(amplitude_list[0][0, 0, 2])
        # subcarrier_phase_list.append(phase_list[0][0, 0, 2])
        subcarrier_amplitude_list.append(amplitude_list[0])
        #subcarrier_phase_list.append(phase_list[0][0, 0, 2])
        amplitude_list.pop(0)
        #phase_list.pop(0)
        subcarrier_amplitude_list.pop(0)
    #print("subcarrier_amplitude_list",np.array(subcarrier_amplitude_list).shape)
    subcarrier_amplitude_list_filter=np.array(subcarrier_amplitude_list).reshape((500,270))
    subcarrier_amplitude_list_filter_one=pre_handle_onlysmooth(subcarrier_amplitude_list_filter[:,0:1])
    #print("subcarrier_amplitude_list_filter",subcarrier_amplitude_list_filter.shape)
    #subcarrier_phase_list_filter = smooth(hampel(np.array(subcarrier_phase_list)), 25)
    subcarrier_amplitude_list_filter_one = np.array(subcarrier_amplitude_list_filter_one)
    #subcarrier_phase_list_filter = np.array(subcarrier_phase_list_filter)
    # subcarrier_amplitude_list_filter = (
    #                                            subcarrier_amplitude_list_filter - subcarrier_amplitude_list_filter.mean()) / subcarrier_amplitude_list_filter.std()
    # subcarrier_phase_list_filter = (
    #                                        subcarrier_phase_list_filter - subcarrier_phase_list_filter.mean()) / subcarrier_phase_list_filter.std()
    # subcarrier_phase_list_filter = subcarrier_phase_list_filter.tolist()
    #subcarrier_amplitude_list_filter = subcarrier_amplitude_list_filter.tolist()

    #print("我们的滤波数组形状" + str(subcarrier_amplitude_list_filter_one.shape))
    #print("保留数组的长度" + str(len(subcarrier_amplitude_list)))
    subcarrier_amplitude_list_filter_one=data_cut_threshold(subcarrier_amplitude_list_filter_one)
    if type(subcarrier_amplitude_list_filter_one) != bool:
        subcarrier_amplitude_list_filter_all=np.array(pre_handle_onlysmooth(subcarrier_amplitude_list_filter))
        subcarrier_amplitude_list_filter_all=data_cut_threshold(subcarrier_amplitude_list_filter_all)
        #print("我们的截取后滤波数组形状" + str(subcarrier_amplitude_list_filter_all.shape))
        result=model_test_on(subcarrier_amplitude_list_filter_all, model_path)
        if(result==0):
            pre="检测到跌倒动作"
            code=1
        if (result == 1):
            pre="检测到非跌倒动作"
            code=0
        print(pre)
    else:
        print("没有检测到活动")
        pre ="没有检测到活动"
        code = 0

    data = {"code": code, "time": time.localtime(time.time()), "predict": pre,"realtime":1}

    print()
    print()

    data_json = json.dumps(data)
    i = 0
    tem = len(conns_pool)
    #print("conns_pool",len(conns_pool))
    while i < tem:
        try:
            send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            conns_pool.pop(i)
            tem = len(conns_pool)
        i = i + 1

