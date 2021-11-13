import numpy as np
def data_cut(data,sampleNum=200):
    all_index = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        max_var = 0
        max_index = 0
        for j in range(data.shape[0]-sampleNum):
            current_var=data[j:j+sampleNum,i:i+1].var
            if(current_var>=max_var):
                max_var=current_var
                max_index=j
        all_index[i]=max_index
    mean_index=np.mean(all_index)
    return data[mean_index:mean_index+sampleNum]

