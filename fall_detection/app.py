import os

import paramiko
from flask import Flask,request
from function.parameter import *
from function.communication import communicate
# from function.collect import *
import json
import socket
from threading import Thread
import csv
from function.service import *
from function.manage_file import *

app = Flask(__name__)

get_file_thread = None
filename = ''
path = ''



@app.route('/connect_rx')
def connect_rx():
    # 接收器的详细的信息
    print("connect_rx已调用")
    # 创建传输文件的trans对象
    if connectType == 1:
        global trans
        try:
            trans = paramiko.Transport((RX_ip, RX_port))
            # 连接接收器
            ssh.connect(hostname=RX_ip, port=RX_port, username=RX_username, password=RX_password)
            ssh1.connect(hostname=RX_ip, port=RX_port, username=RX_username, password=RX_password)
            #ssh.exec_command(RX_FLIE+"/ping.sh")
            # ssh.exec_command(RX_FLIE+"/monitor_wifi_on.sh")
            # time.sleep(1)
            #ssh.exec_command(RX_FLIE + "/monitor_wifi_on.sh")
            time.sleep(0.2)
            #ssh.exec_command(RX_FLIE + "/monitor_wifi_on.sh")
            trans.connect(username=RX_username, password=RX_password)
        except paramiko.ssh_exception.SSHException:
            return json.dumps({"code": 0, "msg": "timeout"})
        except RuntimeError:
            return json.dumps({"code": 2,"msg":"runtimeError"})
        return json.dumps({"code": 1,"msg":"success"})
    elif connectType == 2:
        try:
            trans = paramiko.Transport((RX_ip, RX_port))
            # 连接接收器
            ssh.connect(hostname=RX_ip, port=RX_port, username=RX_username, password=RX_password)
            #ssh.exec_command(RX_FLIE+"/monitor_wifi_on.sh")
            trans.connect(username=RX_username, password=RX_password)
        except paramiko.ssh_exception.SSHException:
            return json.dumps({"code": 0, "msg": "timeout"})
        except RuntimeError:
            return json.dumps({"code": 2,"msg":"runtimeError"})
        return json.dumps({"code": 1,"msg":"success"})

#关闭和小电脑的连接
@app.route('/disconnect_rx')
def disconnect_rx():
    if connectType == 1:
        print("disconnect_rx已调用")
        trans.close()
        ssh.close()
        ssh1.close()
        return json.dumps({"code": 1,"msg":"success"})
    if connectType == 2:
        print("disconnect_rx已调用")
        trans.close()
        ssh.close()
        return json.dumps({"code": 1,"msg":"success"})

#和前端建立socket连接
@app.route('/get_socket')
def get_socket():
    print("get_socket调用")
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        #sock.bind(("192.168.31.173", 8887))
        #sock.bind(("192.168.1.117", 8887))
        sock.bind((sock_ip, 8887))
    except OSError:
        pass
    sock.listen(5)
    socket_thread = Thread(target=communicate, args=(sock, conns_pool))
    socket_thread.start()


    time.sleep(5)
    print("conns_pool", len(conns_pool))
    return json.dumps({"code": 1, "msg": "建立socket连接"})

#数据采集
@app.route('/collect_data')
def collect_data():
    print("collect_data已调用")
    #label为0代表跌倒动作，为1代表其他动作
    global label
    global filename
    global path

    filename = request.args.get("filename")
    path = request.args.get("path")
    label = request.args.get("label")

    global topath
    global frompath
    dirPath = os.path.join(data_dir,path)

    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    topath = os.path.join(data_dir,path, filename + ".dat")
    frompath = RX_FLIE +"/"+ filename + ".dat"
    if connectType == 1:
        #global get_file_thread
        try:
            ssh.exec_command("du -s " + frompath)
            ssh1.exec_command("du -s " + frompath)
            ssh1.exec_command(RX_FLIE + "/ping_3000.sh")
            time.sleep(1)
            #ssh1.exec_command("du -s " + frompath)
            ssh.exec_command(RX_FLIE+"/log.sh " + filename)

        except AttributeError:
            return json.dumps({"code": 0, "msg": "开始采集失败"})
    elif connectType == 2:
        try:
            ssh.exec_command("du -s " + frompath)
            time.sleep(0.5)
            ssh.exec_command(RX_FLIE+"/log.sh " + filename)
        except AttributeError:
            return json.dumps({"code": 0, "msg": "采集失败"})
    return json.dumps({"code": 1, "msg": "正在采集数据"})


#数据采集停止
@app.route('/stop_collect')
def stop_collect():
    print("stop_collect已调用")

    if connectType == 1:
        ssh.exec_command("du -s " + frompath)
        sftp = paramiko.SFTPClient.from_transport(trans)
        try:
            print("frompath:",frompath)
            print("topath:",topath)
            sftp.get(frompath, topath)
        except IOError:
            return json.dumps({"code": 0, "msg": "停止采集失败"})
        ssh1.exec_command("rm " + frompath)

    elif connectType == 2:
        ssh.exec_command("du -s " + frompath)
        sftp = paramiko.SFTPClient.from_transport(trans)
        try:
            print("frompath:",frompath)
            print("topath:",topath)
            sftp.get(frompath, topath)
        except IOError:
            return json.dumps({"code": 0, "msg": "停止采集失败"})
        ssh.exec_command("rm " + frompath)

    filename_write = filename+'.dat'
    ##记录文件名与坐标的映射关系
    with open(label_path, 'a+',newline='') as f:
        csv_write = csv.writer(f)
        data = []
        data.append(filename_write)
        data.append(label)
        csv_write.writerow(data)

    # get_file_thread = None
    return json.dumps({"code": 1, "msg": "停止采集成功"})


#用于拉取小电脑的文件
@app.route('/stop')
def stop_activity():
    filename = request.args.get("filename")
    ssh.exec_command("du -s " + RX_FLIE+"/"+filename+".dat")
    sftp = paramiko.SFTPClient.from_transport(trans)
    try:
        print(filename)
        sftp.get(RX_FLIE+"/"+filename+".dat", "C:\\Users\\11029\\Desktop\\"+filename+".dat")
    except IOError:
        pass
    #ssh.exec_command("rm " + frompath)
    #get_file_thread = None
    return json.dumps({"code": 1, "msg": "停止采集"})

#数据预处理
@app.route('/pre_data')
def pre_data():
    print("pre_data已调用")

    data_path = request.args.get("data_path")
    data_path_pre=data_path+"_pre"
    dirPath = os.path.join(data_dir, data_path)
    if not os.path.exists(dirPath):
        return json.dumps({"code": 0, "msg": "数据文件夹不存在"})
    dirPath_pre=os.path.join(data_dir, data_path+"_pre")
    if not os.path.exists(dirPath_pre):
        os.makedirs(dirPath_pre)
    dataList = os.listdir(dirPath)
    filename = "data_model_dir//map//states.json"
    with open(filename, 'r') as f:
        line = f.readline()
        states = json.loads(line)
    states['pre_data'] = 2  # 设置状态为预处理中
    with open(filename, 'w+') as f:
        json.dump(states, f)
    global predata_thread
    if predata_thread[0] == 0:
        try:
            predata_thread[0] = Thread(target=pre_datalist,args=(dataList,dirPath,dirPath_pre,data_path,data_path_pre))
            predata_thread[0].start()
        except:
            stop_thread(predata_thread[0])
            predata_thread[0]=0
        # if(train_thread[0] == 0):
        #     return json.dumps({"code": 1, "msg": "训练完成"})
        # train_stop()
        return json.dumps({"code": 1, "msg": "数据正在预处理"})
    else:
        return json.dumps({"code": 0, "msg": "已有预处理在进行中"})

#模型训练
@app.route('/train')
def train_model():
    print("train已调用")

    model_name = request.args.get("model_name")
    # storage_path = request.args.get("storage_path")

    dirName = request.args.get("dirname")
    # data_list = json.loads(data_list)["data_list"]
    print(dirName)
    dirPath = os.path.join(data_dir,dirName)
    global model_path
    model_path = os.path.join(model_dir, model_name) + ".h5"
    print(model_name, dirPath, type(dirPath))

    #保存最新的模型名，定位时读取最后一行模型名
    with open(model_nama_path,'a') as f:
        f.write(model_name +'.h5'+ '\n')

    filename = "data_model_dir//map//states.json"
    with open(filename,'r') as f:
        line = f.readline()
        states = json.loads(line)
    states['modelTrain'] = 2#设置状态为训练中
    with open(filename, 'w+') as f:
        json.dump(states, f)

    global train_thread
    if train_thread[0] == 0:
        train_thread[0] = Thread(target=model_train,
                                 args=(dirPath, model_path,dirName,model_name+".h5"))
        train_thread[0].start()
        # if(train_thread[0] == 0):
        #     return json.dumps({"code": 1, "msg": "训练完成"})
        #train_stop()
        return json.dumps({"code": 1, "msg": "开始训练"})
    else:
        return json.dumps({"code": 0, "msg": "已有模型正在训练中"})


@app.route('/train_stop')
def train_stop():
    print("train_stop已调用")
    filename = "data_model_dir//map//states.json"
    with open(filename, 'r') as f:
        line = f.readline()
        states = json.loads(line)
    states['modelTrain'] = 1  # 设置状态为训练完成
    with open(filename, 'w+') as f:
        json.dump(states, f)

    global train_thread
    if(train_thread[0] == 0):
        if not os.path.exists(model_path):
            return json.dumps({"code": 0, "msg": "训练停止，模型保存失败"})
        else:
            return json.dumps({"code": 1, "msg": "已停止训练，模型保存成功"})
    else:
        stop_thread(train_thread[0])
        train_thread[0] = 0
        K.clear_session()
        print("已停止训练")
        if not os.path.exists(model_path):
            return json.dumps({"code": 0, "msg": "训练停止，模型保存失败"})
        else:
            return json.dumps({"code": 1, "msg": "已停止训练，模型保存成功"})


#数据测试
@app.route('/fall_detect')
def test_model():
    print("已调用fall_detect")
    filename_test = "test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    topath = os.path.join(test_dir, filename_test + ".dat")
    frompath = RX_FLIE+"/" + filename_test + ".dat"

    with open(model_nama_path,'r') as f:
        data = f.readlines()
        model_name = data[-1].replace('\n','')
    model_path = os.path.join(model_dir,model_name)
    global get_file_thread

    if connectType ==1:
        print("连接方式1")

        try:
            ssh1.exec_command(RX_FLIE+"/ping.sh")
            time.sleep(1)
            ssh.exec_command(RX_FLIE+"/log.sh " + filename_test)
        except AttributeError:
            return json.dumps({"code": 0, "msg": "实时检测失败"})
        filename = "data_model_dir//map//states.json"
        with open(filename, 'r') as f:
            line = f.readline()
            states = json.loads(line)
        states['fall_detect'] = 2  # 设置状态为实时监测中
        with open(filename, 'w+') as f:
            json.dump(states, f)
        if not get_file_thread:
            get_file_thread = Thread(target=get_file, args=(trans, frompath, topath,model_path))
            get_file_thread.start()
            print("已经开启实时检测")
            return json.dumps({"code": 1, "msg": "已经开启实时检测"})

        else:
            return json.dumps({"code": 2, "msg": "正在采集中,请稍后重试"})

    elif connectType == 2:
        print("连接方式2")
        try:
            ssh.exec_command(RX_FLIE + "/log.sh " + filename_test)
        except AttributeError:
            return json.dumps({"code": 0, "msg": "实时检测失败"})
        filename = "data_model_dir//map//states.json"
        with open(filename, 'r') as f:
            line = f.readline()
            states = json.loads(line)
        states['fall_detect'] = 2  # 设置状态为实时监测中
        with open(filename, 'w+') as f:
            json.dump(states, f)
        if not get_file_thread:
            get_file_thread = Thread(target=get_file1, args=(trans, frompath, topath, model_path))
            get_file_thread.start()
            print("已经开启实时检测")
            return json.dumps({"code": 1, "msg": "已经开启实时检测"})

        else:
            return json.dumps({"code": 2, "msg": "正在采集中,请稍后重试"})

#跌倒实时检测停止
@app.route('/fall_detect_stop')
def realtime_stop_activity():
    print("fall_detect_stop")
    global get_file_thread
    if connectType == 1:
        ssh.exec_command("du -s " + RX_FLIE+"/test.dat")
        stop_thread(get_file_thread)
        sftp = paramiko.SFTPClient.from_transport(trans)
        sftp.get(RX_FLIE+"/test.dat", os.path.join(test_dir,"test.dat"))
        ssh1.exec_command("rm " + RX_FLIE+"/test.dat")
        filename = "data_model_dir//map//states.json"
        with open(filename, 'r') as f:
            line = f.readline()
            states = json.loads(line)
        states['fall_detect'] = 1  # 设置状态为关闭实时监测
        with open(filename, 'w+') as f:
            json.dump(states, f)
        get_file_thread = None
        return json.dumps({"code": 1, "msg": "已经关闭实时监测"})
    elif connectType == 2:
        ssh.exec_command("du -s " + RX_FLIE + "/test.dat")
        stop_thread(get_file_thread)
        sftp = paramiko.SFTPClient.from_transport(trans)
        sftp.get(RX_FLIE + "/test.dat", os.path.join(test_dir, "test.dat"))
        ssh.exec_command("rm " + RX_FLIE + "/test.dat")
        filename = "data_model_dir//map//states.json"
        with open(filename, 'r') as f:
            line = f.readline()
            states = json.loads(line)
        states['fall_detect'] = 1  # 设置状态为关闭实时监测
        with open(filename, 'w+') as f:
            json.dump(states, f)
        get_file_thread = None
        return json.dumps({"code": 1, "msg": "已经关闭实时监测"})
    return json.dumps({"code": 1, "msg": "已经关闭实时监测"})

@app.route('/fall_detect_2')
def test_model_2():
    print("已调用fall_detect")
    filename_test = "test_2"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    topath = os.path.join(test_dir, filename_test + ".dat")
    frompath = RX_FLIE+"/" + filename_test + ".dat"

    with open(model_nama_path,'r') as f:
        data = f.readlines()
        model_name = data[-1].replace('\n','')
    model_path = os.path.join(model_dir,model_name)
    global get_file_thread
    if connectType ==1:
        print("连接方式1")

        try:
            ssh1.exec_command(RX_FLIE+"/ping.sh")
            time.sleep(1)
            ssh.exec_command(RX_FLIE+"/log.sh " + filename_test)
        except AttributeError:
            return json.dumps({"code": 0, "msg": "实时检测失败"})
        if not get_file_thread:
            print("正在采集测试数据")
        else:
            return json.dumps({"code": 2, "msg": "正在采集中,请稍后重试"})

    elif connectType == 2:
        print("连接方式2")
        try:
            ssh.exec_command(RX_FLIE + "/log.sh " + filename_test)
        except AttributeError:
            return json.dumps({"code": 0, "msg": "实时检测失败"})
        if not get_file_thread:
            print("正在采集测试数据")
        else:
            return json.dumps({"code": 2, "msg": "正在采集中,请稍后重试"})

    time.sleep(10)
    # 判断test文件是否存在，若存在 则删除
    if os.path.exists(topath):
        os.remove(topath)
    print("正在停止数据采集")
    if connectType == 1:
        ssh.exec_command("du -s " + frompath)
        sftp = paramiko.SFTPClient.from_transport(trans)
        try:
            sftp.get(frompath, topath)
        except IOError:
            return json.dumps({"code": 0, "msg": "停止采集失败"})
        ssh1.exec_command("rm " + frompath)

    elif connectType == 2:
        ssh.exec_command("du -s " + frompath)
        sftp = paramiko.SFTPClient.from_transport(trans)
        try:
            sftp.get(frompath, topath)
        except IOError:
            return json.dumps({"code": 0, "msg": "停止采集失败"})
        ssh.exec_command("rm " + frompath)


    result = model_test(topath,model_path)#0或者1，0表示跌倒，1代表无事发生
    if(result==0):
        return json.dumps({"code":0,"msg":"检测到跌倒动作"})
    if (result == 1):
        return json.dumps({"code":1,"msg":"检测到非跌倒动作"})
    if (result == -1):
        return json.dumps({"code":-1,"msg":"未检测到活动"})

    # filename = "data_model_dir//map//states.json"
    # with open(filename,'r') as f:
    #     line = f.readline()
    #     states = json.loads(line)
    # states['fall_detect'] = 1 #设置状态为定位结束
    # with open(filename, 'w+') as f:
    #     json.dump(states, f)
    # if(result==0):
    #     return json.dumps({"code":0,"msg":"检测到跌倒"})
    # if (result == 1):
    #     return json.dumps({"code":1,"msg":"未检测到跌倒"})


#获得我们的文件系统信息
@app.route('/getfiles')
def get_files():    
    dataset_path = data_dir
    model_path = model_dir
    data = {"code": 1, "dataset_path": dataset_path, "model_path": model_path,
            "item_num": len(get_all_file_in_dir(dataset_path)),
            "datatype_num": len(get_one_dir_in_dir(dataset_path)),
            "model_num": len(get_all_file_in_dir(model_path)),
            "dataset_list": get_information_in_dir(dataset_path),
            "model_list": get_information_in_dir(model_path)}
    return json.dumps(data)

#获得文件信息
@app.route('/filemanagement/getfileinfo')
def get_file_information(path="/home/zhw/dataset_model/model"):
    path = request.args.get("path")
    data = {"code": 1, "current_path": path, "child": get_information_in_dir(path)}
    return json.dumps(data)

#删除文件
@app.route('/filemanagement/deletefile')
def deletefile():
    paths = request.args.get("paths")
    print(paths)
    paths = json.loads(paths)["paths"]
    if delete_file(paths):
        return json.dumps({"code": 1})
    else:
        return json.dumps({"code": 0})

#删除文件夹
@app.route('/filemanagement/deletedir')
def deletedir():
    path = request.args.get("path")
    if delete_all_in_dir(path):
        os.rmdir(path)
        return json.dumps({"code": 1})
    else:
        return json.dumps({"code": 0})

#创建文件夹
@app.route('/filemanagement/createdir')
def createdir():
    filename = request.args.get("filename")
    parent_path = request.args.get("parent_path")
    path = parent_path + "/" + filename
    if create_dir(path):
        return json.dumps({"code": 1, "child": get_information_in_dir(parent_path)})
    else:
        return json.dumps({"code": 0})

#生成文件树
@app.route('/filemanagement/tree')
def showtree():
    dir_path = "data_model_dir/data_dir"
    model_path = "data_model_dir/model_dir"

    data_dir = get_child_information(dir_path)
    data_model = get_child_tofile_information(model_path)
    #print("data_dir", data_dir)
    #print("data_model",data_model)

    len_node=len(data_dir["nodes"])
    print(len_node)
    data_root=[]
    data_pre = []
    for i in range(len_node):
        if("_pre" in data_dir["nodes"][i]["text"]):
            data_pre.append(data_dir["nodes"][i]["text"])
        else:
            data_root.append(data_dir["nodes"][i]["text"])
    print(data_root)
    print(data_pre)
    return json.dumps({"code": 1, "data_root": data_root, "data_pre":data_pre,"data_model": data_model})


@app.route('/getstates')
def get_states():
    filename = "data_model_dir//map//states.json"
    with open(filename,'r') as f:
        line = f.readline()
        states = json.loads(line)
    data = {'code':1,
            'modelTrain':states['modelTrain'],
            "fall_detect":states['fall_detect'],
            "pre_data":states['pre_data']}
    return json.dumps(data)


if __name__ == '__main__':
    app.run('0.0.0.0', port=8889, debug=True)
