import paramiko
import numpy as np
# 这是一个参数列表用于存储我们这个系统的各个模块功能使用的参数

# 这是app模块的要用到和传递给各个模块的参数


get_file_thread = None  # 这是我们获取文件的线程标识符，我们将进程的内存地址给他
train_thread = [0]  # 这是我们训练的线程标识符，我们将进程的内存地址给他
predata_thread=[0]
test_thread=[0]  # 这是我们训练的线程标识符，我们将进程的内存地址给他
conns_pool = []  # 这是socket连接池

trans = None  # 用于保存paramiko.Transport的对象
# 创建SSH对象
ssh = paramiko.SSHClient()
ssh1 = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# frompath = '/home/rtlab420/1.dat'  # 最初始化的发射器路径
# topath = '/home/zhw/1.dat'  # 最初是化的服务器文件路径

# 服务器的相关连接配置
sock_ip="192.168.1.117"
# 接收器的相关连接配置
RX_ip = "192.168.1.113"
RX_port = 22
RX_username = "tangkai"
RX_password = "root"

#RX_ip = "192.168.0.114"
#RX_username = "rtlab420"
connectType = 1 #1为路由器连接 2为小电脑连接
sampleNum=300
collectSleepTime = 30
testSleepTime = 10

##保存原始数据和模型的目录
data_dir = "data_model_dir\\data_dir"
model_dir = "data_model_dir\\model_dir"
test_dir = "data_model_dir\\test_dir"
#RX_FLIE="/home/rtlab420"
#命令执行目录
RX_FLIE="/home/tangkai"

# 这是service要用到和传递给各个模块的参数
#subcarrier_amplitude_list = [0 for index in range(400)]  # 描述振幅载波初始化
subcarrier_amplitude_list = np.zeros((500, 3, 3, 30),dtype=object).tolist()
subcarrier_phase_list = [0 for i in range(500)]  # 描述相位载波初始化
subcarrier_phase_list_filter = []  # 描述振幅过滤载波初始化
subcarrier_amplitude_list_filter = []  # 描述相位过滤载波初始化
distinguish_list=np.zeros((500, 3, 3, 30)) #这个用来存放全部原始数据n*3*3*30格式，用于识别
time_end = 0  # 描述采集的当前时间，和time_start相减成为当前采集时间

#这是service的参数配置
model_nama_path="data_model_dir\map\model_name.txt"
label_path = 'data_model_dir\map\labels.csv'

#网络参数
dimension = 270
batch_size = 32
epochs = 100