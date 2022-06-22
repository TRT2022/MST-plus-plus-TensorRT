#         ====TensorRT Hackathon 2022   Medcare AI Lab====
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#    

'''
   MST++ TensorRT INT8量化的banchmark和精度计算和可视化

'''

import os
import argparse
import ctypes
from glob import glob 
import architecture
import torch
import onnxruntime

import numpy as np
import heapq
import pandas as pd
import tensorrt as trt
from cuda import cudart

import cv2
from time import time_ns,sleep

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('ggplot')

files = os.listdir("./data")
soFileList = glob("./" + "*.so")


# about time
def time_static_torch(input_data,model_path='./model/mst_plus_plus.pth',batch_size=1,nRound=1000):
    #pytorch
    torch_model =  architecture.MST_Plus_Plus() # 导入模型
    check_point = torch.load(model_path,map_location='cpu')
    # torch_model.load_state_dict(check_point['state_dict'])  # 初始化权重
    torch_model.load_state_dict({k.replace('module.', ''): v for k, v in check_point['state_dict'].items()},
                                strict=True)
    # torch_model.half()
    torch_model.eval()
    torch_model.to("cuda")

    torch_input = torch.from_numpy(input_data).cuda()
    
    for i in range(50):
        with torch.no_grad():
            torch_output = torch_model(torch_input)

    torch.cuda.synchronize()
    t0 = time_ns()
    for i in range(nRound):
        with torch.no_grad():
            torch_output = torch_model(torch_input)
        # torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t1 = time_ns()

    Latency_pytorch = (t1 - t0)/1000/1000/nRound
    Throughput_pytorch = 1000/Latency_pytorch*batch_size

    # 清空释放显存
    del torch_model
    torch_input.cpu()
    del torch_input
    torch.cuda.empty_cache()

    return Latency_pytorch, Throughput_pytorch


def time_static_trt(input_data,plan_path,batch_size=1,nRound=1000):

    with open(plan_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()

    inputH0 = np.ascontiguousarray(input_data.ravel())
    _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    outputH0 = np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

    for i in range(50):
        context.execute_v2([int(inputD0),int(outputD0)])

    torch.cuda.synchronize()
    t0 = time_ns()
    for i in range(nRound):
        context.execute_v2([int(inputD0),int(outputD0)])
    torch.cuda.synchronize()
    t1 = time_ns()

    Latency_trt = (t1 - t0)/1000/1000/nRound
    Throughput_trt = 1000/Latency_trt*batch_size

    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

    return Latency_trt, Throughput_trt



# about accuracy

def acc_static_torch(input,model):

    torch_input = torch.from_numpy(input).cuda()
    with torch.no_grad():
        torch_output = model(torch_input)

    return torch_output.cpu().detach().numpy()[:, :, 128:-128, 128:-128]


def acc_static_trt(input,engine,context):

    inputH0 = np.ascontiguousarray(input.ravel())
    _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    outputH0 = np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

    context.execute_v2([int(inputD0),int(outputD0)])

    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)
    
    return outputH0[:, :, 128:-128, 128:-128]


def cut_dat(x,n=20):
    '''
        为了保持统计的合理性，去掉前5个最大值和最小值在做统计
    '''
    x_index = map(x.index,heapq.nlargest(n, x))
    y_index = map(x.index,heapq.nsmallest(n, x))
    for i in list(x_index)+list(y_index):
        x[i] = 0.0
    return x


if __name__ == "__main__":
    #pytorch
    torch_model =  architecture.MST_Plus_Plus() # 导入模型
    check_point = torch.load("./model/mst_plus_plus.pth",map_location='cpu')
    # torch_model.load_state_dict(check_point['state_dict'])  # 初始化权重
    torch_model.load_state_dict({k.replace('module.', ''): v for k, v in check_point['state_dict'].items()},
                                strict=True)
    # torch_model.half()
    torch_model.eval()
    torch_model.to("cuda")

    # trt
    logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.VERBOSE

    trt.init_libnvinfer_plugins(logger, '')
    PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

    for soFile in soFileList:
       ctypes.cdll.LoadLibrary(soFile)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(logger)
    network = builder.create_network(EXPLICIT_BATCH)

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30

    #load engine
    if os.path.exists("./model/mst_plus_plus_onnxparser_b1_int8.plan"):
        # If a serialized engine exists, load it instead of building a new one.
        with open("./model/mst_plus_plus_onnxparser_b1_int8.plan", "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading %s"%decoderPlanFile)
            exit()

    context = engine.create_execution_context()


    # time

    batch_sizes = [8,4,2,1]

    # FP32
    static_torch_32 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}
    static_trt_onnxparser_int8 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}    

    for batch_size in batch_sizes:
        print(f"[INFO] 当前测试的batch size为：{batch_size}")
        np.random.seed(2022)
        input_data = np.random.uniform(-1,1,(batch_size, 3, 512, 482)).astype(np.float32)

        # torch
        print("[INFO] 正在进行pytorch测试")
        l_torch,t_torch = time_static_torch(input_data=input_data,model_path='./model/mst_plus_plus.pth',batch_size=batch_size,nRound=1000)
        sleep(10)


        # int8 trt onnxparser
        print("[INFO] 正在进行trt onnxparser int8测试")
        l_trt_1,t_trt_1 = time_static_trt(input_data=input_data,plan_path=f"./model/mst_plus_plus_onnxparser_b{batch_size}_int8.plan",batch_size=batch_size,nRound=1000)
        lsu_1 = round(l_torch / l_trt_1,2)
        tsu_1 = round(t_trt_1 / t_torch ,2)
        sleep(10)

      
        #pytorch
        static_torch_32['batch_size'].append(batch_size)
        static_torch_32['latency'].append(l_torch)
        static_torch_32['throughput'].append(t_torch)
        static_torch_32['LSU'].append("1x")
        static_torch_32['TSU'].append("1x")

        #trt onnxparser
        static_trt_onnxparser_int8['batch_size'].append(batch_size)
        static_trt_onnxparser_int8['latency'].append(l_trt_1)
        static_trt_onnxparser_int8['throughput'].append(t_trt_1)
        static_trt_onnxparser_int8['LSU'].append(str(lsu_1)+"x")
        static_trt_onnxparser_int8['TSU'].append(str(tsu_1)+"x")

    print("="*50)
    print("torch")
    print(static_torch_32)
    print("int8")
    print(static_trt_onnxparser_int8)
    print("="*50)
    

    # acc
    files = os.listdir("./data")

    abserror_onnxparser_int8 = []
    relerror_onnxparser_int8 = []
    for file in files:
        # print(file)
        file_path = os.path.join("./data/",file)

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_fp32 = np.array(img,dtype=np.float32)
        img_fp32 = (img_fp32 - img_fp32.min()) / (img_fp32.max() - img_fp32.min())

        img_1 = img_fp32.transpose([2,1,0]).reshape(1,3,512,482)  # 3x512x482

        torch_output =  acc_static_torch(img_1,model=torch_model)
        onnxparser_int8_output = acc_static_trt(img_1,engine,context)


        # 绝对误差 计算指标
        abserror_onnxparser_int8.append( np.mean(np.abs(torch_output-onnxparser_int8_output)) )

        # 相对误差 计算指标
        relerror_onnxparser_int8.append( np.mean(np.abs(torch_output-onnxparser_int8_output) /(np.abs(torch_output)+1e-12) ) )


    # mean abs 
    plt.rcParams['figure.figsize'] = (16.0, 9.0) # 单位是inches
    fig,subs=plt.subplots(1,2)

    subs[0].plot(np.arange(len(files)), cut_dat(abserror_onnxparser_int8), 'bv',label='TensorRT ONNXParser (INT8)')
    subs[0].set_xlabel('Test Image ID')
    subs[0].set_ylabel("Absolute Error")
    subs[0].legend(edgecolor='blue')

    # mean relative abs
    subs[1].plot(np.arange(len(files)), cut_dat(relerror_onnxparser_int8), 'ro',label='TensorRT ONNXParser (INT8)')
    subs[1].set_xlabel('Test Image ID')
    subs[1].set_ylabel("Relative Error")
    subs[1].legend(edgecolor='blue')


    col_labels = ['Mean','Median','Max']
    row_labels = ['Absolute Error',"Relative Error"]
    table_vals = [[round(np.mean(cut_dat(abserror_onnxparser_int8)),8),round(np.median(cut_dat(abserror_onnxparser_int8)),8),round(np.max(cut_dat(abserror_onnxparser_int8)),8)],
        [round(np.mean(cut_dat(relerror_onnxparser_int8)),8),round(np.median(cut_dat(relerror_onnxparser_int8)),8),round(np.max(cut_dat(relerror_onnxparser_int8)),8)]
    ]

    row_colors = ['red','gold','green',"blue","purple"]
    table = subs[1].table(cellText=table_vals, colWidths=[0.1]*3,
                        rowLabels=row_labels, colLabels=col_labels,
                        rowColours=row_colors, colColours=row_colors,
                        bbox=(-0.75, -0.2, 1.65, 0.1))
    table.auto_set_font_size(False)
    table.set_fontsize(15)


    plt.savefig("./int8_acc.png",bbox_inches='tight', pad_inches=0)
    plt.close()


   
