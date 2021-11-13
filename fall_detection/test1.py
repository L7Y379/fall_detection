from threading import Thread
import json
from flask import Flask,request

import socket

from function.communication import communicate
from function.parameter import *
from testThread import *
app = Flask(__name__)


@app.route('/hello')
def hellotest():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", 8887))
    except OSError:
        pass
    sock.listen(5)
    # 等待用户连接
    socket_thread = Thread(target=communicate, args=(sock, conns_pool))
    socket_thread.start()
    return json.dumps({"code": 0, "msg": "hello success"})

@app.route('/test')
def test():
    test_thread1 = Thread(target=testT)
    test_thread1.start()
    return json.dumps({"code": 0, "msg": "testthread success"})

if __name__ == '__main__':
    app.run('0.0.0.0', port=8888, debug=True)


