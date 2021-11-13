import json
from function.parameter import *
from function.communication import *

def testT():
    print(111)
    print(222)
    print(333)
    print(444)
    print(111)
    print(111)
    print(666)


    data = {"code": 1, "type": 5, "acc": 1 / 2}
    data_json = json.dumps(data)
    i = 00
    tem = len(conns_pool)
    while i < tem:
        try:
            send_msg(conns_pool[i], bytes(data_json, encoding="utf-8"))
        except BrokenPipeError:
            conns_pool.pop(i)
            tem = len(conns_pool)
        i = i + 1
