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
   MST++ TensorRT 绝对误差和相对误差的平均值，中位数，最大值计算和可视化

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
if os.path.exists("./model/mst_plus_plus_b1.plan"):
    # If a serialized engine exists, load it instead of building a new one.
    with open("./model/mst_plus_plus_b1.plan", "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading %s"%decoderPlanFile)
        exit()

context = engine.create_execution_context()

#load engine
if os.path.exists("./model/mst_plus_plus_onnxparser_b1.plan"):
    # If a serialized engine exists, load it instead of building a new one.
    with open("./model/mst_plus_plus_onnxparser_b1.plan", "rb") as f:
        engine_1 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine_1 == None:
        print("Failed loading %s"%decoderPlanFile)
        exit()

context_1 = engine_1.create_execution_context()

if os.path.exists("./model/mst_plus_plus_b1_fp16.plan"):
    # If a serialized engine exists, load it instead of building a new one.
    with open("./model/mst_plus_plus_b1_fp16.plan", "rb") as f:
        engine_2 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine_2 == None:
        print("Failed loading %s"%decoderPlanFile)
        exit()

context_2 = engine_2.create_execution_context()

# load engien 2
#load engine
if os.path.exists("./model/mst_plus_plus_onnxparser_b1_fp16.plan"):
    # If a serialized engine exists, load it instead of building a new one.
    with open("./model/mst_plus_plus_onnxparser_b1_fp16.plan", "rb") as f:
        engine_3 = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine_3 == None:
        print("Failed loading %s"%decoderPlanFile)
        exit()

context_3 = engine_3.create_execution_context()


#onnxruntime
ort_session = onnxruntime.InferenceSession("./model/mst_plus_plus_b1.onnx",providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']


def acc_static_torch(input,model=torch_model):

    torch_input = torch.from_numpy(input).cuda()
    with torch.no_grad():
        torch_output = model(torch_input)

    return torch_output.cpu().detach().numpy()[:, :, 128:-128, 128:-128]


def acc_static_onnx(input,model=ort_session):
    ort_inputs = {"input":input}
    output_name = ort_session.get_outputs()[0].name

    output = model.run([output_name],ort_inputs)

    return output[0][:, :, 128:-128, 128:-128]



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

    # 平均绝对误差的计算
    # np.mean(np.abs(output_data_pytorch - output_data_trt)
    # 平均相对精度的计算
    # np.mean(np.abs(output_data_pytorch - output_data_trt) / np.abs(output_data_pytorch))

    # 只在途中画出相对绝对误差的 最大值，最小值，和中位数
    files = os.listdir("./data")

    abserror_onnx = []
    abserror_onnxparser_32 = []
    abserror_api_32 = []
    abserror_api_16 = []
    abserror_onnxparser_16 = []
    abserror_onnxparser_16 = []
    

    relerror_onnx = []
    relerror_onnxparser_32 = []
    relerror_api_32 = []
    relerror_onnxparser_16 = []
    relerror_api_16 = []
    relerror_onnxparser_16 = []



    for file in files:
        print(file)
        file_path = os.path.join("./data/",file)

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_fp32 = np.array(img,dtype=np.float32)
        img_fp32 = (img_fp32 - img_fp32.min()) / (img_fp32.max() - img_fp32.min())

        img_1 = img_fp32.transpose([2,1,0]).reshape(1,3,512,482)  # 3x512x482

        torch_output =  acc_static_torch(img_1)
        onnx_output = acc_static_onnx(img_1)

        onnxparser_32_output = acc_static_trt(img_1,engine_1,context_1)
        onnxparser_16_output = acc_static_trt(img_1,engine_3,context_3)

        api_32_output = acc_static_trt(img_1,engine,context)
        api_16_output = acc_static_trt(img_1,engine_2,context_2)

        # 绝对误差 计算指标
        abserror_onnx.append( np.mean(np.abs(torch_output-onnx_output)) )
        abserror_onnxparser_32.append( np.mean(np.abs(torch_output-onnxparser_32_output)) )
        abserror_api_32.append( np.mean(np.abs(torch_output-api_32_output)) )
        abserror_onnxparser_16.append( np.mean(np.abs(torch_output-onnxparser_16_output)) )
        abserror_api_16.append( np.mean(np.abs(torch_output-api_16_output)) )


        # 相对误差 计算指标
        relerror_onnx.append( np.mean(np.abs(torch_output-onnx_output) / (np.abs(torch_output)+1e-12)) )
        relerror_onnxparser_32.append( np.mean(np.abs(torch_output-onnxparser_32_output) /(np.abs(torch_output)+1e-12) ) )
        relerror_api_32.append( np.mean(np.abs(torch_output-api_32_output) /(np.abs(torch_output)+1e-12) ) )
        relerror_onnxparser_16.append( np.mean(np.abs(torch_output-onnxparser_16_output) /(np.abs(torch_output)+1e-12) ) )
        relerror_api_16.append( np.mean(np.abs(torch_output-api_16_output) /(np.abs(torch_output)+1e-12) ) )
       
  

    # Category = ["onnxruntime"] * (len(abserror_onnx))+["TensorRT ONNXParser(FP32)"] * len(abserror_onnxparser_32) + ["TensorRT API(FP32)"]*len(abserror_api_32) +["TensorRT ONNXParser(FP16)"] * len(abserror_onnxparser_16) + ["TensorRT API(FP16)"] * len(abserror_api_16)

    # abs_error = {'Category':Category,
    #         'Absolute Error':cut_dat(abserror_onnx) + cut_dat(abserror_onnxparser_32) + cut_dat(abserror_api_32) + cut_dat(abserror_onnxparser_16)+cut_dat(abserror_api_16),
    #         'File Index': list(range(len(files)))*5}

    # df = pd.DataFrame(abs_error)
    # sns.jointplot(data=df, x='File Index', y='Absolute Error', hue='Category')

    # col_labels = ['Mean','Median','Max']
    # row_labels = ['onnxruntime','ONNXParser(FP32)','TensorRT API(FP32)','ONNXParser(FP16)','TensorRT API(FP16)']
    # table_vals = [[np.mean(cut_dat(abserror_onnx)),np.max(cut_dat(abserror_onnx)),np.min(cut_dat(abserror_onnx))],
    #     [round(np.mean(cut_dat(abserror_onnxparser_32)),9),np.max(cut_dat(abserror_onnxparser_32)),np.min(cut_dat(abserror_onnxparser_32))],
    #     [round(np.mean(cut_dat(abserror_api_32)),9),np.max(cut_dat(abserror_api_32)),np.min(cut_dat(abserror_api_32))],
    #     [round(np.mean(cut_dat(abserror_onnxparser_16)),9),np.max(cut_dat(abserror_onnxparser_16)),np.min(cut_dat(abserror_onnxparser_16))],
    #     [round(np.mean(cut_dat(abserror_api_16)),9),np.max(cut_dat(abserror_api_16)),np.min(cut_dat(abserror_api_16))]]
    # row_colors = ['red','gold','green',"blue","purple"]
    # table = plt.table(cellText=table_vals, #colWidths=[0.1]*8,
    #                      rowLabels=row_labels, colLabels=col_labels,
    #                      rowColours=row_colors, colColours=row_colors,
    #                      bbox=(-3, 0.6, 2, 0.2))
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)

    # plt.savefig("./mean_abs_error.png",bbox_inches='tight', pad_inches=0)
    # plt.close()

    # mean abs 
    plt.rcParams['figure.figsize'] = (16.0, 9.0) # 单位是inches
    fig,subs=plt.subplots(2,3)
    subs[0][0].plot(np.arange(len(files)), cut_dat(abserror_onnx), 'ro',label='onnxruntime')
    # subs[0][0].axhline(y=np.max(mean_abserror_onnx),color='b',linestyle='--')
    # subs[0][0].axhline(y=np.min(mean_abserror_onnx),color='b',linestyle='--')
    # subs[0][0].axhline(y=np.min(mean_abserror_onnx),color='b',linestyle='--')
    subs[0][0].set_xlabel('Test Image ID')
    subs[0][0].set_ylabel("Absolute Error")
    subs[0][0].legend(edgecolor='blue')

    subs[0][1].plot(np.arange(len(files)), cut_dat(abserror_onnxparser_32), 'bv',label='TensorRT ONNXParser (FP32)')
    # subs[0][1].axhline(y=np.max(mean_abserror_onnxparser_32),color='b',linestyle='--')
    # subs[0][1].axhline(y=np.min(mean_abserror_onnxparser_32),color='b',linestyle='--')
    subs[0][1].set_xlabel('Test Image ID')
    subs[0][1].set_ylabel("Absolute Error")
    subs[0][1].legend(edgecolor='blue')

    subs[0][2].plot(np.arange(len(files)), cut_dat(abserror_api_32), 'g^',label='TensorRT API (FP32)')
    # subs[0][2].axhline(y=np.max(mean_abserror_api_32),color='b',linestyle='--')
    # subs[0][2].axhline(y=np.min(mean_abserror_api_32),color='b',linestyle='--')
    subs[0][2].set_xlabel('Test Image ID')
    subs[0][2].set_ylabel("Absolute Error")
    subs[0][2].legend(edgecolor='blue')

    subs[1][0].plot(np.arange(len(files)), cut_dat(abserror_onnxparser_16), 'k*',label='TensorRT ONNXParser (FP16)')
    # subs[1][0].axhline(y=np.max( mean_abserror_onnxparser_16),color='b',linestyle='--')
    # subs[1][0].axhline(y=np.min( mean_abserror_onnxparser_16),color='b',linestyle='--')
    subs[1][0].set_xlabel('Test Image ID')
    subs[1][0].set_ylabel("Absolute Error")
    subs[1][0].legend(edgecolor='blue')

    subs[1][1].plot(np.arange(len(files)), cut_dat(abserror_api_16), 'mh',label='TensorRT API (FP16)')
    # subs[1][1].axhline(y=np.max(mean_abserror_api_16),color='b',linestyle='--')
    # subs[1][1].axhline(y=np.min(mean_abserror_api_16),color='b',linestyle='--')
    subs[1][1].set_xlabel('Test Image ID')
    subs[1][1].set_ylabel("Absolute Error")
    subs[1][1].legend(edgecolor='blue')

    # plt.delaxes(subs[1][2])
    plt.delaxes()

    col_labels = ['Mean','Median','Max']
    row_labels = ['onnxruntime','ONNXParser(FP32)','TensorRT API(FP32)','ONNXParser(FP16)','TensorRT API(FP16)']
    table_vals = [[round(np.mean(cut_dat(abserror_onnx)),8),round(np.median(cut_dat(abserror_onnx)),8),round(np.max(cut_dat(abserror_onnx)),8)],
        [round(np.mean(cut_dat(abserror_onnxparser_32)),8),round(np.median(cut_dat(abserror_onnxparser_32)),8),round(np.max(cut_dat(abserror_onnxparser_32)),8)],
        [round(np.mean(cut_dat(abserror_api_32)),8),round(np.median(cut_dat(abserror_api_32)),8),round(np.max(cut_dat(abserror_api_32)),8)],
        [round(np.mean(cut_dat(abserror_onnxparser_16)),8),round(np.median(cut_dat(abserror_onnxparser_16)),8),round(np.max(cut_dat(abserror_onnxparser_16)),8)],
        [round(np.mean(cut_dat(abserror_api_16)),8),round(np.median(cut_dat(abserror_api_16)),8),round(np.max(cut_dat(abserror_api_16)),8)]]
    row_colors = ['red','gold','green',"blue","purple"]
    table = subs[1][1].table(cellText=table_vals, colWidths=[0.1]*3,
                        rowLabels=row_labels, colLabels=col_labels,
                        rowColours=row_colors, colColours=row_colors,
                        bbox=(-0.75, -0.7, 2.9, 0.5))
    table.auto_set_font_size(False)
    table.set_fontsize(13)



    plt.savefig("./abs_error.png",bbox_inches='tight', pad_inches=0)
    plt.close()


    # mean relative abs
    plt.rcParams['figure.figsize'] = (16.0, 9.0) # 单位是inches
    fig,subs=plt.subplots(2,3)
    subs[0][0].plot(np.arange(len(files)), cut_dat(relerror_onnx), 'ro',label='onnxruntime')
    # subs[0][0].axhline(y=np.max(mean_abserror_onnx),color='b',linestyle='--')
    # subs[0][0].axhline(y=np.min(mean_abserror_onnx),color='b',linestyle='--')
    # subs[0][0].axhline(y=np.min(mean_abserror_onnx),color='b',linestyle='--')


    subs[0][0].set_xlabel('Test Image ID')
    subs[0][0].set_ylabel("Relative Error")
    subs[0][0].legend(edgecolor='blue')

    subs[0][1].plot(np.arange(len(files)), cut_dat(relerror_onnxparser_32), 'bv',label='TensorRT ONNXParser (FP32)')
    # subs[0][1].axhline(y=np.max(mean_abserror_onnxparser_32),color='b',linestyle='--')
    # subs[0][1].axhline(y=np.min(mean_abserror_onnxparser_32),color='b',linestyle='--')
    subs[0][1].set_xlabel('Test Image ID')
    subs[0][1].set_ylabel("Relative Error")
    subs[0][1].legend(edgecolor='blue')

    subs[0][2].plot(np.arange(len(files)), cut_dat(relerror_api_32), 'g^',label='TensorRT API (FP32)')
    # subs[0][2].axhline(y=np.max(mean_abserror_api_32),color='b',linestyle='--')
    # subs[0][2].axhline(y=np.min(mean_abserror_api_32),color='b',linestyle='--')
    subs[0][2].set_xlabel('Test Image ID')
    subs[0][2].set_ylabel("Relative Error")
    subs[0][2].legend(edgecolor='blue')

    subs[1][0].plot(np.arange(len(files)), cut_dat(relerror_onnxparser_16), 'k*',label='TensorRT ONNXParser (FP16)')
    # subs[1][0].axhline(y=np.max( mean_abserror_onnxparser_16),color='b',linestyle='--')
    # subs[1][0].axhline(y=np.min( mean_abserror_onnxparser_16),color='b',linestyle='--')
    subs[1][0].set_xlabel('Test Image ID')
    subs[1][0].set_ylabel("Relative Error")
    subs[1][0].legend(edgecolor='blue')

    subs[1][1].plot(np.arange(len(files)), cut_dat(relerror_api_16), 'mh',label='TensorRT API (FP16)')
    # subs[1][1].axhline(y=np.max(mean_abserror_api_16),color='b',linestyle='--')
    # subs[1][1].axhline(y=np.min(mean_abserror_api_16),color='b',linestyle='--')
    subs[1][1].set_xlabel('Test Image ID')
    subs[1][1].set_ylabel("Relative Error")
    subs[1][1].legend(edgecolor='blue')

    # plt.delaxes(subs[1][2])
    plt.delaxes()

    col_labels = ['Mean','Median','Max']
    row_labels = ['onnxruntime','ONNXParser(FP32)','TensorRT API(FP32)','ONNXParser(FP16)','TensorRT API(FP16)']
    table_vals = [[round(np.mean(cut_dat(relerror_onnx)),8),round(np.median(cut_dat(relerror_onnx)),8),round(np.max(cut_dat(relerror_onnx)),8)],
        [round(np.mean(cut_dat(relerror_onnxparser_32)),8),round(np.median(cut_dat(relerror_onnxparser_32)),8),round(np.max(cut_dat(relerror_onnxparser_32)),8)],
        [round(np.mean(cut_dat(relerror_api_32)),8),round(np.median(cut_dat(relerror_api_32)),8),round(np.max(cut_dat(relerror_api_32)),8)],
        [round(np.mean(cut_dat(relerror_onnxparser_16)),8),round(np.median(cut_dat(relerror_onnxparser_16)),8),round(np.max(cut_dat(relerror_onnxparser_16)),8)],
        [round(np.mean(cut_dat(relerror_api_16)),8),round(np.median(cut_dat(relerror_api_16)),8),round(np.max(cut_dat(relerror_api_16)),8)]]
    row_colors = ['red','gold','green',"blue","purple"]
    table = subs[1][1].table(cellText=table_vals, colWidths=[0.1]*3,
                        rowLabels=row_labels, colLabels=col_labels,
                        rowColours=row_colors, colColours=row_colors,
                        bbox=(-0.75, -0.7, 2.9, 0.5))
    table.auto_set_font_size(False)
    table.set_fontsize(13)



    plt.savefig("./rel_error.png",bbox_inches='tight', pad_inches=0)
    plt.close()