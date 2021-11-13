import os
import time

#这个函数输入的是我们的路径信息，输出的是这个路径的所有的文件的名字的一个列表
def get_all_file_in_dir(dir_path):
    '''
    获取目录下的所有文件
    :return:
    '''
    file_name_list = []
    for root, dirs, files in os.walk(dir_path):
        if files:
            for name in files:
                # 此处可以增加文件名称过滤条件， 比如 -----
                # 跳过所有 ~$ 开头的文件
                # if name.startswith(('~$',)):
                #     continue
                # ---------------------------------------

                file_name = '{0}/{1}'.format(root, name).replace('\\', '/')
                file_name_list.append(file_name)
    return file_name_list

#这个函数输入的是一个路径，输出的是该路径下的第一层的所有目录（仅仅是第一层的文件夹）
def get_one_dir_in_dir(dir_path):
    dir_list = []
    for element in os.listdir(dir_path):
        element_path = os.path.join(dir_path, element)
        if os.path.isdir(element_path):
            dir_list.append(element_path)
    return dir_list

#这个函数输入路径名字，输出的是该路径下所有的文件夹和文件的信息，包括创建时间、路径、名字，对于文件夹还包括了这个文件夹
#下的文件夹数目和这个文件夹的所有文件总数
def get_information_in_dir(dir_path):
    list = []
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            x = {"name": item, "path": item_path, "item_num": len(get_all_file_in_dir(item_path)),
                 "folder_num": len(get_one_dir_in_dir(item_path)),
                 "createTime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(item_path).st_mtime))}
            list.append(x)
        else:
            x = {"name": item, "path": item_path,
                 "createTime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(item_path).st_mtime))}
            list.append(x)
    return list

#删除文件，对于一个路径列表删除这个列表中包含的所有路径的文件
def delete_file(list):
    try:
        for element in list:
            os.remove(element)
    except OSError:
        return False
    return True

#删除一个文件夹的所有文件和文件夹
def delete_all_in_dir(path):
    try:
        for element in os.listdir(path):
            element_path = os.path.join(path, element)
            if os.path.isdir(element_path):
                delete_all_in_dir(element_path)
                os.rmdir(element_path)
                print("文件夹已经删除：" + element_path)
            else:
                os.remove(element_path)
    except OSError:
        return False
    return True

#创建文件夹
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False

#获得一个文件夹的文件夹子项的信息，递归操作从而找到该文件夹所有文件夹子项的信息，输出的信息作为一个json的传递给前端
def get_child_information(selfpath):
    data = {"text": selfpath.split('/')[-1], "path": selfpath}
    for item in os.listdir(selfpath):
        #item_path = os.path.join(selfpath, item)
        item_path=selfpath+"/"+item
        if os.path.isdir(item_path):
            if "nodes" in data.keys():
                data["nodes"].append(get_child_information(item_path))
            else:
                data["nodes"] = [get_child_information(item_path)]
    return data


#获得一个文件夹的所有子项（包括文件夹和文件），输出信息作为json的传递给前端
def get_child_tofile_information(selfpath):
    data = {"text": selfpath.split('/')[-1], "path": selfpath}
    for item in os.listdir(selfpath):
        item_path = os.path.join(selfpath, item)
        if os.path.isdir(item_path):
            if "nodes" in data.keys():
                data["nodes"].append(get_child_tofile_information(item_path))
            else:
                data["nodes"] = [get_child_tofile_information(item_path)]
        else:
            if "nodes" in data.keys():
                data["nodes"].append({"text": item, "path": item_path})
            else:
                data["nodes"] = [{"text": item, "path": item_path}]

    return data