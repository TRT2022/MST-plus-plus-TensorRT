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
mst++ benchmark的统计和可视化包括：Latency， ThroughPut, 加速比等（pytorch, onnxruntime, TensorRT ONNXParser, TensorRT API)
'''

import os
import argparse
import ctypes
from glob import glob 
import torch
import architecture
import onnxruntime

import numpy as np
import tensorrt as trt
from cuda import cudart

from time import time_ns,sleep

# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('ggplot')

files = os.listdir("./data")
soFileList = glob("./" + "*.so")

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


def time_static_onnx(input_data,onnx_path,batch_size=1,nRound=1000):
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_inputs = {"input":input_data}
    output_name = ort_session.get_outputs()[0].name


    
    for i in range(50):
        output = ort_session.run([output_name],ort_inputs)
    
    torch.cuda.synchronize()
    t0 = time_ns()
    for i in range(nRound):
        output = ort_session.run(None,ort_inputs)
    torch.cuda.synchronize()
    t1 = time_ns()

    Latency_onnx = (t1 - t0)/1000/1000/nRound
    Throughput_onnx = 1000/Latency_onnx*batch_size

    return Latency_onnx, Throughput_onnx
    


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



if __name__ == "__main__":
    # # Latency and Throughput
    # Pytorch batch size = 32时 out of memory, 因此我们仅对比了batch size是16以下的batch

    batch_sizes = [16,8,4,2,1]

    # FP32
    static_torch_32 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}
    static_trt_onnx_32 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}
    static_trt_onnxparser_32 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}
    static_trt_onnxparser_16 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}    
    static_trt_api_32 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}
    static_trt_api_16 = {'batch_size':[],'latency':[],'throughput':[],'LSU':[],"TSU":[]}

    for batch_size in batch_sizes:
        print(f"[INFO] 当前测试的batch size为：{batch_size}")
        np.random.seed(2022)
        input_data = np.random.uniform(-1,1,(batch_size, 3, 512, 482)).astype(np.float32)

        # torch
        print("[INFO] 正在进行pytorch测试")
        l_torch,t_torch = time_static_torch(input_data=input_data,model_path='./model/mst_plus_plus.pth',batch_size=batch_size,nRound=1000)
        sleep(10)

        # onnxruntime
        if batch_size == 16:  # batch_size=16
            l_onnx = 0
            t_onnx = 0
            lsu_onnx = 1
            tsu_onnx = 1
        else:
            print("[INFO] 正在进行onnxruntime测试")
            l_onnx,t_onnx = time_static_onnx(input_data=input_data,onnx_path=f'./model/mst_plus_plus_b{batch_size}.onnx',batch_size=batch_size,nRound=1000)
            lsu_onnx = round( l_torch / l_onnx,2)
            tsu_onnx = round( t_onnx/ t_torch ,2)
            sleep(10)

        # fp32 trt api
        print("[INFO] 正在进行trt api FP32测试")
        l_trt,t_trt = time_static_trt(input_data=input_data,plan_path=f"./model/mst_plus_plus_b{batch_size}.plan",batch_size=batch_size,nRound=1000)
        lsu = round( l_torch / l_trt,2)
        tsu = round( t_trt / t_torch ,2)
        sleep(10)

        # fp32 trt onnxparser
        print("[INFO] 正在进行trt onnxparser FP32测试")
        l_trt_1,t_trt_1 = time_static_trt(input_data=input_data,plan_path=f"./model/mst_plus_plus_onnxparser_b{batch_size}.plan",batch_size=batch_size,nRound=1000)
        lsu_1 = round(l_torch / l_trt_1,2)
        tsu_1 = round(t_trt_1 / t_torch ,2)
        sleep(10)

        # fp16 trt api
        print("[INFO] 正在进行trt api FP16测试")
        l_trt_16,t_trt_16 = time_static_trt(input_data=input_data,plan_path=f"./model/mst_plus_plus_onnxparser_b{batch_size}_fp16.plan",batch_size=batch_size,nRound=1000)
        lsu_16 = round(l_torch /l_trt_16,2)
        tsu_16 = round(t_trt_16 /t_torch,2)
        sleep(10)

        # fp16 trt onnxparser
        print("[INFO] 正在进行trt onnxparser FP16测试")
        l_trt_16_1,t_trt_16_1 = time_static_trt(input_data=input_data,plan_path=f"./model/mst_plus_plus_b{batch_size}_fp16.plan",batch_size=batch_size,nRound=1000)
        lsu_16_1 = round(l_torch / l_trt_16_1,2)
        tsu_16_1 = round(t_trt_16_1 / t_torch,2)
        sleep(10)

        #pytorch
        static_torch_32['batch_size'].append(batch_size)
        static_torch_32['latency'].append(l_torch)
        static_torch_32['throughput'].append(t_torch)
        static_torch_32['LSU'].append("1x")
        static_torch_32['TSU'].append("1x")
        #onnx
        static_trt_onnx_32['batch_size'].append(batch_size)
        static_trt_onnx_32['latency'].append(l_onnx)
        static_trt_onnx_32['throughput'].append(t_onnx)
        static_trt_onnx_32['LSU'].append(str(lsu_onnx)+"x")
        static_trt_onnx_32['TSU'].append(str(tsu_onnx)+"x")

        # trt api
        static_trt_api_32['batch_size'].append(batch_size)
        static_trt_api_32['latency'].append(l_trt)
        static_trt_api_32['throughput'].append(t_trt)
        static_trt_api_32['LSU'].append(str(lsu)+"x")
        static_trt_api_32['TSU'].append(str(tsu)+"x")

        static_trt_api_16['batch_size'].append(batch_size)
        static_trt_api_16['latency'].append(l_trt_16)
        static_trt_api_16['throughput'].append(t_trt_16)
        static_trt_api_16['LSU'].append(str(lsu_16)+"x")
        static_trt_api_16['TSU'].append(str(tsu_16)+"x")

        #trt onnxparser
        static_trt_onnxparser_32['batch_size'].append(batch_size)
        static_trt_onnxparser_32['latency'].append(l_trt_1)
        static_trt_onnxparser_32['throughput'].append(t_trt_1)
        static_trt_onnxparser_32['LSU'].append(str(lsu_1)+"x")
        static_trt_onnxparser_32['TSU'].append(str(tsu_1)+"x")

        static_trt_onnxparser_16['batch_size'].append(batch_size)
        static_trt_onnxparser_16['latency'].append(l_trt_16_1)
        static_trt_onnxparser_16['throughput'].append(t_trt_16_1)
        static_trt_onnxparser_16['LSU'].append(str(lsu_16_1)+"x")
        static_trt_onnxparser_16['TSU'].append(str(tsu_16_1)+"x")


    print("-"*50)
    print("torch:")
    print(static_torch_32)
    print("onnxruntime")
    print(static_trt_onnx_32)
    print("trt api fp32:")
    print(static_trt_api_32)
    print("trt api fp16:")
    print(static_trt_api_16)
    print("trt onnxparser fp32:")
    print(static_trt_onnxparser_32)
    print("trt onnxparser fp16:")
    print(static_trt_onnxparser_16)
    print("-"*50)

    # plot latency vs throughput
    torch_x = static_torch_32['latency']
    torch_y = static_torch_32['throughput']

    onnx_x = static_trt_onnx_32['latency']
    onnx_y = static_trt_onnx_32['throughput']

    trt_api_32_x = static_trt_api_32['latency']
    trt_api_32_y = static_trt_api_32['throughput']

    trt_api_16_x = static_trt_api_16['latency']
    trt_api_16_y = static_trt_api_16['throughput']

    trt_onnx_32_x = static_trt_onnxparser_32['latency']
    trt_onnx_32_y = static_trt_onnxparser_32['throughput']

    trt_onnx_16_x = static_trt_onnxparser_16['latency']
    trt_onnx_16_y = static_trt_onnxparser_16['throughput']

    plt.rcParams['figure.figsize'] = (16.0, 9.0)
    # pytorch
    plt.plot(torch_x, torch_y, 'ro--',label='Pytorch',markersize=7,linewidth=2)
    for i,(a, b) in enumerate(zip(torch_x, torch_y)):
        # plt.text(a+15,b-0.15,'(%d,%d,%d)'%(batch_size[i],a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'r'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'r'})

    # onnx
    plt.plot(onnx_x[1:], onnx_y[1:], 'ms:',label='onnxruntime',markersize=7,linewidth=2)
    for i,(a, b) in enumerate(zip(onnx_x[1:], onnx_y[1:])):
        # plt.text(a+15,b-0.15,'(%d,%d,%d)'%(batch_size[i],a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'r'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i+1]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'm'})

    # trt api 
    plt.plot(trt_api_32_x, trt_api_32_y, 'b^-',label='TensorRT API (FP32)',markersize=7,linewidth=2)
    for i,(a, b) in enumerate(zip(trt_api_32_x, trt_api_32_y)):
        # plt.text(a+15,b-0.15,'(%d,%d)'%(a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'b'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'b'})


    plt.plot(trt_api_16_x, trt_api_16_y, 'g*-.',label='TensorRT API (FP16)',markersize=7,linewidth=2)
    for i,(a, b) in enumerate(zip(trt_api_16_x, trt_api_16_y)):
        # plt.text(a+15,b-0.15,'(%d,%d)'%(a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'g'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'g'})
        if batch_sizes[i] in [4,8]:
            plt.annotate(f"({int(a)},{int(b)})",xy=(a,b),xytext=(a*0.9,b*0.9),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

    # trt onnxparser
    plt.plot(trt_onnx_32_x, trt_onnx_32_y, 'yp--',label='TensorRT ONNXParser (FP32)',markersize=7,linewidth=2)
    for i,(a, b) in enumerate(zip(trt_onnx_32_x, trt_onnx_32_y)):
        # plt.text(a+15,b-0.15,'(%d,%d)'%(a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'b'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'y'})


    plt.plot(trt_onnx_16_x, trt_onnx_16_y, 'kh-.',label='TensorRT ONNXParser (FP16)',markersize=7,linewidth=2)
    for i,(a, b) in enumerate(zip(trt_onnx_16_x, trt_onnx_16_y)):
        # plt.text(a+15,b-0.15,'(%d,%d)'%(a,b),ha='center', va='bottom',fontdict={'size': 10, 'color':  'g'})
        plt.text(a+15,b-0.15,f'Batch:{batch_sizes[i]}',ha='center', va='bottom',fontdict={'size': 10, 'color':  'k'})
        if batch_sizes[i] in [4,8]:
            plt.annotate(f"({int(a)},{int(b)})",xy=(a,b),xytext=(a*0.9,b*0.9),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
    
    
    plt.xlabel('Latency (ms)')
    plt.ylabel( r"$Throughput=\frac{1000}{Latency} \times Batch\_Size$")
    plt.legend()
    plt.savefig("./latency_vs_throughput.png",bbox_inches='tight', pad_inches=0)
    plt.close()
