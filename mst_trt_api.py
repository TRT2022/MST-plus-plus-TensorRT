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

使用TensorRT API重新搭建 MST++网络并进行加速！！！ 支持 FP32,FP16,INT8

'''

import os
import ctypes
from glob import glob 
import numpy as np
import argparse
import configparser
import tensorrt as trt
# import pycuda.autoinit
# import pycuda.driver as cuda
from cuda import cudart


import warnings
warnings.filterwarnings("ignore")


logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.VERBOSE

# plugin creater
trt.init_libnvinfer_plugins(logger, '')
layernorm = ctypes.CDLL("./LayerNormPlugin.so",mode=ctypes.RTLD_GLOBAL)
# gemm = ctypes.CDLL("./CuBLASGemmPlugin.so",mode=ctypes.RTLD_GLOBAL)
normalize = ctypes.CDLL("./NormalizePlugin.so",mode=ctypes.RTLD_GLOBAL)


plg_registry = trt.get_plugin_registry()
ln_plg_creator = plg_registry.get_plugin_creator("LayerNorm", "1", "")
# gemm_creator = plg_registry.get_plugin_creator("CuBLASGemm", "1",  "")
norm_creator = plg_registry.get_plugin_creator("Normalize", "1",  "")

PLUGIN_CREATORS = plg_registry.plugin_creator_list

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

builder = trt.Builder(logger)
network = builder.create_network(EXPLICIT_BATCH)

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30


def torch_normalize(network,inputs,weights=None,name="L2Norm"):
    '''inputs : input list, 1 element
        weights: weights list, 0 elemnt

    '''
    # reduce_l2  axes=-1, keepdims=1  会被拆分为 sqrt(sum( pow(x,2) ))
    # pow(2)
    # pow2_0 = network.add_scale(input=inputs[0],mode=trt.ScaleMode.UNIFORM,shift=np.array([0.0],dtype=np.float32),scale=np.array([1.0],dtype=np.float32),power=np.array([2],dtype=np.float32))
    # pow2_0.name="normalize_pow"
    pow_shape = inputs[0].shape
    pow_data = np.ones(pow_shape,dtype=np.float32)*2
    pow_val = network.add_constant(pow_shape, pow_data)
    pow2_0 = network.add_elementwise(input1 = inputs[0],input2=pow_val.get_output(0),op=trt.ElementWiseOperation.POW)
    pow2_0.name = f"{name}_pow2"

    # sum
    reduce_l2_0 = network.add_reduce(input=pow2_0.get_output(0),op=trt.ReduceOperation.SUM,axes=1<<3, keep_dims=1)  # 第4个维度规约？
    reduce_l2_0.name = f"{name}_ReduceL2"
    #sqrt
    sqrt_shape = reduce_l2_0.get_output(0).shape
    sqrt_data = np.ones(sqrt_shape,dtype=np.float32)*0.5
    sqrt_val = network.add_constant(sqrt_shape, sqrt_data)
    # sqrt_val = network.add_constant((1,1,1,1), np.array([[[[0.5]]]],dtype=np.float32))
    sqrt_0 = network.add_elementwise(input1 = reduce_l2_0.get_output(0),input2=sqrt_val.get_output(0),op=trt.ElementWiseOperation.POW)
    sqrt_0.name = f"{name}_Sqrt"

    # pow2_1 = network.add_scale(input=reduce_l2_0.get_output(0),mode=trt.ScaleMode.UNIFORM,shift=np.array([0.0],dtype=np.float32),scale=np.array([1.0],dtype=np.float32),power=np.array([0.5],dtype=np.float32))
    # # clip_0 = network.add_activation(input=pow2_1.get_output(0), type=trt.ActivationType.CLIP)
    # # clip_0.alpha = -np.inf
    # # clip_0.beta = 1e-12
    clip_val = network.add_constant((1,1,1,1), np.array([[[[1e-12]]]],dtype=np.float32))
    clip_0 = network.add_elementwise(input1 = sqrt_0.get_output(0),input2=clip_val.get_output(0),op=trt.ElementWiseOperation.MAX)
    # div
    div_0 = network.add_elementwise(input1 = inputs[0],input2=clip_0.get_output(0),op=trt.ElementWiseOperation.DIV)
    div_0.name = f"{name}_Div"

    return network,div_0



# [N, C, H, W]
def torch_normalize_plugin():
    '''调用了TensorRT 官方的版本，发现有问题
       https://github.com/NVIDIA/TensorRT/issues/2020

    '''
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == "Normalize":
            acrossSpatial = trt.PluginField("acrossSpatial", np.array([1],np.int32),trt.PluginFieldType.INT32)
            channelShared  = trt.PluginField("channelShared", np.array([1],np.int32),trt.PluginFieldType.INT32)
            eps = trt.PluginField("eps", np.array([1e-14],dtype=np.float32) , trt.PluginFieldType.FLOAT32)
            weights = trt.PluginField("weights", np.array([1.0],dtype=np.float32) , trt.PluginFieldType.FLOAT32)
            nbweights = trt.PluginField("nbWeights", np.array([1],dtype=np.int32) , trt.PluginFieldType.INT32)
            plugin_version = trt.PluginField("plugin_version", np.array(["1"],dtype=np.string_), trt.PluginFieldType.CHAR)

            field_collection = trt.PluginFieldCollection([acrossSpatial,channelShared,eps,weights,nbweights,plugin_version])
            plugin = plugin_creator.create_plugin(name="Normalize", field_collection=field_collection)
    return plugin


# def gemm_plugin(weight):
#     parameterList = []
#     parameterList.append(trt.PluginField("weight", np.float32(weight), trt.PluginFieldType.FLOAT32))
#     parameterList.append(trt.PluginField("k", np.int32(weight.shape[2]), trt.PluginFieldType.INT32))
#     parameterList.append(trt.PluginField("n", np.int32(weight.shape[3]), trt.PluginFieldType.INT32))
#     return gemm_creator.create_plugin("CuBLASGemm", trt.PluginFieldCollection(parameterList))


def self_attention(network,K,Q,V,weights,name="SelfAttention"):
    '''softmax(K'Q/sqrt(HeadSize))
       K      : 1x1x31x249856, K matrix
       Q      : 1x1x31x249856, Q matrix
       V      : 1x1x249856x31, V matrix
       weights: weight list

    '''
    # #transpose
    # transpose_0 = network.add_shuffle(K)
    # transpose_0.first_transpose = [0, 1, 3, 2]
    #matmul
    matmul_0 = network.add_matrix_multiply(Q,trt.MatrixOperation.NONE,K,trt.MatrixOperation.NONE)
    matmul_0.op0 = trt.MatrixOperation.NONE
    matmul_0.op1 = trt.MatrixOperation.TRANSPOSE
    matmul_0.name = f"{name}_MatMul"

    # /sqrt(headsize)
    weights_1_val = np.array([weights[0]])
    weights_1_const = network.add_constant(weights_1_val.shape, weights_1_val)
    mul_0 = network.add_elementwise(matmul_0.get_output(0),weights_1_const.get_output(0),trt.ElementWiseOperation.PROD)
    mul_0.name = f"{name}_Mul"

    #softmax
    softmax_0 = network.add_softmax(mul_0.get_output(0))
    softmax_0.axes = 1<<3 #第4维上softmax
    softmax_0.name = f"{name}_SoftMax"

    # V
    transpose_3 = network.add_shuffle(V)
    transpose_3.reshape_dims = weights[1]  # 1x249856x1x31
    transpose_3.second_transpose = [0,2,3,1] # 1x1x31x249856

    transpose_4 = network.add_shuffle(transpose_3.get_output(0))
    transpose_4.reshape_dims = weights[2]
    transpose_4.second_transpose = [0,1,3,2]

    matmul_1 = network.add_matrix_multiply(softmax_0.get_output(0),trt.MatrixOperation.NONE,transpose_3.get_output(0),trt.MatrixOperation.NONE)
    matmul_1.name = f"{name}_V_MatMul"

    #transpose and reshape
    transpose_1 = network.add_shuffle(matmul_1.get_output(0))
    transpose_1.first_transpose = [0, 3, 1, 2]
    transpose_1.reshape_dims = weights[3]

    #matmul
    weights_4_val = np.array([weights[4]])
    weights_4_const = network.add_constant(weights_4_val.shape,weights_4_val)
    matmul_2 = network.add_matrix_multiply(transpose_1.get_output(0),trt.MatrixOperation.NONE,weights_4_const.get_output(0),trt.MatrixOperation.NONE)
    # fc_0 = network.add_fully_connected(transpose_1.get_output(0), 1, params["onnx::MatMul_4337"], params["body.0.bottleneck.blocks.0.0.proj.bias"])
    matmul_2.name = f"{name}_Score_MatMul"

    weights_5_val = np.array([[weights[5]]])
    weights_5_const = network.add_constant(weights_5_val.shape,weights_5_val)
    add_0 = network.add_elementwise(matmul_2.get_output(0),weights_5_const.get_output(0),trt.ElementWiseOperation.SUM)
    add_0.name = f"{name}_Add"

    #reshape
    transpose_2 = network.add_shuffle(add_0.get_output(0))
    transpose_2.reshape_dims = weights[6]

    return network, transpose_2
 


def self_attention_plugin():
    pass


def gelu_plugin(type_id=np.array([0],dtype=np.int32)):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == "CustomGeluPluginDynamic":
            type_id = trt.PluginField("type_id", type_id, trt.PluginFieldType.INT32)
            plugin_version = trt.PluginField("plugin_version", np.array(["1"],dtype=np.string_), trt.PluginFieldType.CHAR)

            field_collection = trt.PluginFieldCollection([type_id,plugin_version])
            plugin = plugin_creator.create_plugin(name="CustomGeluPluginDynamic", field_collection=field_collection)
    return plugin


def layer_norm(network,inputs,weights):
    '''inputs : input list, 1 element
       weights: weight list, 2 element
    '''
    # add_0 = network.add_elementwise(inputs[0],inputs[1],trt.ElementWiseOperation.SUM)
    reducemean_0 = network.add_reduce(inputs[0],op=trt.ReduceOperation.AVG,axes=1<<0,keep_dims=True)
    sub_0 = network.add_elementwise(inputs[0],reducemean_0.get_output(0),trt.ElementWiseOperation.SUB)

    pow_const = network.add_constant((1,1,1,1),np.array([[[[2]]]],dtype=np.float32))
    pow_0 = network.add_elementwise(sub_0.get_output(0),pow_const.get_output(0),trt.ElementWiseOperation.POW)

    reducemean_1 = network.add_reduce(pow_0.get_output(0),op=trt.ReduceOperation.AVG,axes=1<<3,keep_dims=True) # 按照第4个维度求均值

    add_const = network.add_constant((1,1,1,1),np.array([[[[1e-6]]]],dtype=np.float32))
    add_1 = network.add_elementwise(reducemean_1.get_output(0),add_const.get_output(0),trt.ElementWiseOperation.SUM)

    sqrt_const = network.add_constant((1,1,1,1),np.array([[[[0.5]]]],dtype=np.float32))
    sqrt_0 = network.add_elementwise(add_1.get_output(0),sqrt_const.get_output(0),trt.ElementWiseOperation.POW)

    div_0 = network.add_elementwise(sub_0.get_output(0),sqrt_0.get_output(0),trt.ElementWiseOperation.DIV)

    # beta,gamma
    weights_0_val = np.array([[[weights[0]]]],dtype=np.float32).reshape(1,1,-1,1)  # 和batch_size有关？
    weights_0_const = network.add_constant(weights_0_val.shape,weights_0_val)
    matmul_0 = network.add_matrix_multiply(div_0.get_output(0),trt.MatrixOperation.NONE,weights_0_const.get_output(0),trt.MatrixOperation.NONE)
    weights_1_val = np.array([[[weights[1]]]])
    weights_1_const = network.add_constant(weights_1_val.shape,weights_1_val)
    add_2 = network.add_elementwise(matmul_0.get_output(0),weights_1_const.get_output(0),trt.ElementWiseOperation.SUM)

    return network, add_2


def layernorm_plugin():
    # one-flow
    epsilon = trt.PluginField("epsilon", np.array([1e-6]), trt.PluginFieldType.FLOAT32)
    field_collection = trt.PluginFieldCollection([epsilon])
    plugin = ln_plg_creator.create_plugin(name="LayerNorm", field_collection=field_collection)
    return plugin


def s_msa(network,inputs,weights,conv_param,batch_size=1,type_id=np.array([0],dtype=np.int32),name="s_msa"):
    '''inputs : input list, 1 element
       weights: weight
    '''
    # conv_1
    # 
    conv_0 = network.add_convolution(input=inputs[0],num_output_maps=weights[0].shape[0],kernel_shape=(int(conv_param["conv1_kernel"]),int(conv_param["conv1_kernel"])),kernel=weights[0])
    conv_0.stride = (int(conv_param["conv1_strid"]),int(conv_param["conv1_strid"]))
    conv_0.padding = (int(conv_param["conv1_pad"]),int(conv_param["conv1_pad"]))
    conv_0.dilation = (int(conv_param["conv1_dilation"]),int(conv_param["conv1_dilation"]))
    conv_0.num_groups= int(conv_param["conv1_group"])

    # conv_0.precision =trt.DataType.FLOAT
    # # conv_0.reset_precision()
    # print(conv_0.precision)

    #transpose and reshape
    transpose_0 = network.add_shuffle(conv_0.get_output(0))
    transpose_0.first_transpose = [0, 2, 3, 1]
    transpose_0.reshape_dims = weights[1]


    # #matmul
    weights_2_new = np.array([weights[2]])
    weights_2_const = network.add_constant(weights_2_new.shape, weights_2_new)
    matmul_0 = network.add_matrix_multiply(transpose_0.get_output(0),trt.MatrixOperation.NONE,weights_2_const.get_output(0),trt.MatrixOperation.NONE)
    #transpose and reshape
    transpose_1 = network.add_shuffle(matmul_0.get_output(0))
    transpose_1.reshape_dims = weights[3] #1x249856x1x31
    # transpose_1.second_transpose = [1,0,2,3]  #[N,c,h,w] 249856x1x1x31
    transpose_1.second_transpose = [0,2,3,1]  #1x1x31x249856 #换回来

    # K
    # norm_0 = network.add_plugin_v2(inputs=[transpose_1.get_output(0)], plugin=torch_normalize_plugin())  #<-------------
    # #1x1x249856x31 
    # transpose_2 = network.add_shuffle(norm_0.get_output(0))  # 249856x1x1x31
    # transpose_2.first_transpose = [1,2,3,0]  #[N,c,h,w] 1x1x31x249856

    network,norm_0 = torch_normalize(network,[transpose_1.get_output(0)],name=f"{name}_K_L2Norm") #换回来

    # Q
    weights_4_new = np.array([weights[4]])
    weights_4_const = network.add_constant(weights_4_new.shape, weights_4_new)
    matmul_1 = network.add_matrix_multiply(transpose_0.get_output(0),trt.MatrixOperation.NONE,weights_4_const.get_output(0),trt.MatrixOperation.NONE)
    matmul_1.name=f"{name}_Q_MatMul"

    # #transpose and reshape
    transpose_3 = network.add_shuffle(matmul_1.get_output(0))
    transpose_3.reshape_dims = weights[5]
    # transpose_3.second_transpose = [1,0,2,3]  #[N,c,h,w] 249856x1x1x31
    transpose_3.second_transpose = [0,2,3,1]  #1x1x31x249856

    # norm_1 = network.add_plugin_v2(inputs=transpose_3.get_output(0), plugin=torch_normalize_plugin())
    # #1x1x249856x31
    # transpose_4 = network.add_shuffle(norm_1.get_output(0))  # 249856x1x1x31
    # transpose_4.first_transpose = [1,2,3,0]  #[N,c,h,w] 1x1x31x249856
    network,norm_1 = torch_normalize(network,[transpose_3.get_output(0)],name=f"{name}_Q_L2Norm")

    # # V
    weights_6_new = np.array([weights[6]])
    weights_6_const = network.add_constant(weights_6_new.shape, weights_6_new)
    matmul_3 = network.add_matrix_multiply(transpose_0.get_output(0),trt.MatrixOperation.NONE,weights_6_const.get_output(0),trt.MatrixOperation.NONE)
    matmul_3.name = f"{name}_V_MatMul"

    # self attention
    weights_3 = weights[7] #list
    # print(len(weights_3))
    network,transpose_4_0 = self_attention(network=network,K=norm_0.get_output(0),Q=norm_1.get_output(0),V=matmul_3.get_output(0),weights=weights_3,name=f"{name}_SelfAttention")  #换回来
    # network,transpose_4_0 = self_attention(network=network,K=transpose_2.get_output(0),Q=norm_1.get_output(0),V=matmul_3.get_output(0),weights=weights_3)


    transpose_5 = network.add_shuffle(matmul_3.get_output(0))
    transpose_5.reshape_dims = weights[8]
    transpose_5.second_transpose = [0,3,1,2]

    #conv_2
    conv_1 = network.add_convolution(input=transpose_5.get_output(0),num_output_maps=weights[9].shape[0],kernel_shape=(int(conv_param["conv2_kernel"]),int(conv_param["conv2_kernel"])),kernel=weights[9])
    conv_1.stride = (int(conv_param["conv2_strid"]),int(conv_param["conv2_strid"]))
    conv_1.padding = (int(conv_param["conv2_pad"]),int(conv_param["conv2_pad"]))
    conv_1.dilation = (int(conv_param["conv2_dilation"]),int(conv_param["conv2_dilation"]))
    conv_1.num_groups= int(conv_param["conv2_group"])
    #gelu  #可能需要unsqueeze一下
    gelu_0 = network.add_plugin_v2(inputs=[conv_1.get_output(0)], plugin=gelu_plugin(type_id))
    #conv_3
    conv_2 = network.add_convolution(input=gelu_0.get_output(0),num_output_maps=weights[10].shape[0],kernel_shape=(int(conv_param["conv3_kernel"]),int(conv_param["conv3_kernel"])),kernel=weights[10])
    conv_2.stride = (int(conv_param["conv3_strid"]),int(conv_param["conv3_strid"]))
    conv_2.padding = (int(conv_param["conv3_pad"]),int(conv_param["conv3_pad"]))
    conv_2.dilation = (int(conv_param["conv3_dilation"]),int(conv_param["conv3_dilation"]))
    conv_2.num_groups= int(conv_param["conv3_group"])
    transpose_6 = network.add_shuffle(conv_2.get_output(0))
    transpose_6.first_transpose = [0,2,3,1]

    # # add
    add_0 = network.add_elementwise(transpose_6.get_output(0),transpose_4_0.get_output(0),trt.ElementWiseOperation.SUM)
    transpose_7 = network.add_shuffle(conv_0.get_output(0))
    transpose_7.first_transpose = [0,2,3,1]

    # add
    add_0_0 = network.add_elementwise(add_0.get_output(0),transpose_7.get_output(0),trt.ElementWiseOperation.SUM)
    # # layer norm
    # inputs_3 = [add_0_0.get_output(0)]
    # weights_5 = weights[11]
    # network,add_1 = layer_norm(network,inputs_3,weights_5)
    # 替换为layernorm plugin one-flow版
    transpose_layernorm = network.add_shuffle(add_0_0.get_output(0))
    transpose_layernorm.reshape_dims = [-1,add_0_0.get_output(0).shape[2],add_0_0.get_output(0).shape[3]]
    layernorm = network.add_plugin_v2(inputs=[transpose_layernorm.get_output(0)],plugin=layernorm_plugin())

    gamma_val = weights[11][0].reshape((1,1,-1))
    gamma = network.add_constant(gamma_val.shape,gamma_val)
    mul_layernorm = network.add_elementwise(layernorm.get_output(0),gamma.get_output(0),trt.ElementWiseOperation.PROD)

    beta_val = weights[11][1].reshape((1,1,-1))
    beta = network.add_constant(beta_val.shape,beta_val)
    add_1 = network.add_elementwise(mul_layernorm.get_output(0),beta.get_output(0),trt.ElementWiseOperation.SUM)

    #transpose
    transpose_8 = network.add_shuffle(add_1.get_output(0))
    transpose_8.reshape_dims = [batch_size,-1,add_1.get_output(0).shape[1],add_1.get_output(0).shape[2]]
    transpose_8.second_transpose = [0,3,1,2]

    #conv_4
    conv_3 = network.add_convolution(input=transpose_8.get_output(0),num_output_maps=weights[12].shape[0],kernel_shape=(int(conv_param["conv4_kernel"]),int(conv_param["conv4_kernel"])),kernel=weights[12])
    conv_3.stride = (int(conv_param["conv4_strid"]),int(conv_param["conv4_strid"]))
    conv_3.padding = (int(conv_param["conv4_pad"]),int(conv_param["conv4_pad"]))
    conv_3.dilation = (int(conv_param["conv4_dilation"]),int(conv_param["conv4_dilation"]))
    conv_3.num_groups= int(conv_param["conv4_group"])
    #gelu  #可能需要unsqueeze一下
    gelu_1 = network.add_plugin_v2(inputs=[conv_3.get_output(0)], plugin=gelu_plugin(type_id))
    #conv_5
    conv_4 = network.add_convolution(input=gelu_1.get_output(0),num_output_maps=weights[13].shape[0],kernel_shape=(int(conv_param["conv5_kernel"]),int(conv_param["conv5_kernel"])),kernel=weights[13])
    conv_4.stride = (int(conv_param["conv5_strid"]),int(conv_param["conv5_strid"]))
    conv_4.padding = (int(conv_param["conv5_pad"]),int(conv_param["conv5_pad"]))
    conv_4.dilation = (int(conv_param["conv5_dilation"]),int(conv_param["conv5_dilation"]))
    conv_4.num_groups= int(conv_param["conv5_group"])
    
    gelu_2 = network.add_plugin_v2(inputs=[conv_4.get_output(0)], plugin=gelu_plugin(type_id))

    #conv_6
    conv_5 = network.add_convolution(input=gelu_2.get_output(0),num_output_maps=weights[14].shape[0],kernel_shape=(int(conv_param["conv6_kernel"]),int(conv_param["conv6_kernel"])),kernel=weights[14])
    conv_5.stride = (int(conv_param["conv6_strid"]),int(conv_param["conv6_strid"]))
    conv_5.padding = (int(conv_param["conv6_pad"]),int(conv_param["conv6_pad"]))
    conv_5.dilation = (int(conv_param["conv6_dilation"]),int(conv_param["conv6_dilation"]))
    conv_5.num_groups= int(conv_param["conv6_group"])

    transpose_9 = network.add_shuffle(conv_5.get_output(0))
    transpose_9.first_transpose = [0,2,3,1]

    # # add
    add_2 = network.add_elementwise(transpose_9.get_output(0),add_0_0.get_output(0),trt.ElementWiseOperation.SUM)

    return network, add_2



def get_block_weights(block_w_cfg,params):
    weights_1 = [params[block_w_cfg["w0"]],params[block_w_cfg["w1"]],params[block_w_cfg["w2"]],
        params[block_w_cfg["w3"]],params[block_w_cfg["w4"]],params[block_w_cfg["w5"]],params[block_w_cfg["w6"]],
        [params[block_w_cfg["w7_1"]],params[block_w_cfg["w7_2"]],params[block_w_cfg["w7_3"]],
        params[block_w_cfg["w7_4"]],params[block_w_cfg["w7_5"]],params[block_w_cfg["w7_6"]],
        params[block_w_cfg["w7_7"]]], 
        params[block_w_cfg["w8"]],params[block_w_cfg["w9"]],params[block_w_cfg["w10"]],
        [params[block_w_cfg["w11_1"]],params[block_w_cfg["w11_2"]]],params[block_w_cfg["w12"]],
        params[block_w_cfg["w13"]],params[block_w_cfg["w14"]] ]
    return weights_1


def mst_plus_plus_trt_api(network,config,params,params_cfg="./mst_config.ini",batch_size=1,type_id=np.array([0],dtype=np.int64)):

    '''mst++由s_msa bloack堆叠而成，其类似于UNet的U型结构，其中共15个s_msa block!!! 

    '''
    
    mst_input = network.add_input("input",trt.DataType.FLOAT,(batch_size,3,512,482))

    padd_0 = network.add_padding_nd(mst_input,(0,0),(0,0))
    padd_0.post_padding = (0,6)

    # 该conv有3个分支
    conv_44 = network.add_convolution_nd(input=padd_0.get_output(0),num_output_maps=31,kernel_shape=(3, 3),kernel=params["conv_in.weight"])
    conv_44.stride = (1,1)
    conv_44.padding = (1,1) #?
    conv_44.dilation = (1,1)
    conv_44.num_groups= 1

    configer_m = configparser.ConfigParser()
    configer_m.read(params_cfg)

    # s_msa_1
    block_1_w_cfg = configer_m['block1_weight']
    block_1_c_cfg = configer_m['block1_conv']


    inputs_1 = [conv_44.get_output(0)]
    weights_1 = get_block_weights(block_1_w_cfg,params)
    network, s_msa_1 = s_msa(network,inputs_1,weights_1,block_1_c_cfg,batch_size,type_id=type_id,name="s_mas1")

    # s_msa_2
    transpose_280 = network.add_shuffle(s_msa_1.get_output(0))
    transpose_280.first_transpose =  [0,3,1,2]
    
    block_2_w_cfg = configer_m['block2_weight']
    block_2_c_cfg = configer_m['block2_conv']

    inputs_2 = [transpose_280.get_output(0)]
    weights_2 = get_block_weights(block_2_w_cfg,params)
    network, s_msa_2 = s_msa(network,inputs_2,weights_2,block_2_c_cfg,batch_size,type_id=type_id,name="s_mas2")

    # s_msa_3
    transpose_516 = network.add_shuffle(s_msa_2.get_output(0))
    transpose_516.first_transpose = [0,3,1,2]

    block_3_w_cfg = configer_m['block3_weight']
    block_3_c_cfg = configer_m['block3_conv']

    inputs_3 = [transpose_516.get_output(0)]
    weights_3 = get_block_weights(block_3_w_cfg,params)
    network, s_msa_3 = s_msa(network,inputs_3,weights_3,block_3_c_cfg,batch_size,type_id=type_id,name="s_mas3")

    # s_msa_4
    transpose_752 = network.add_shuffle(s_msa_3.get_output(0))
    transpose_752.first_transpose = [0,3,1,2]

    convtranspose_753 = network.add_deconvolution(input=transpose_752.get_output(0),num_output_maps=62,kernel_shape=(2,2),kernel=params["body.0.decoder_layers.0.0.weight"],
        bias=params["body.0.decoder_layers.0.0.bias"])
    convtranspose_753.stride=(2,2)
    convtranspose_753.padding=(0,0)
    convtranspose_753.num_groups=1

    concat_754 = network.add_concatenation([convtranspose_753.get_output(0),transpose_516.get_output(0)])
    concat_754.axis = 1

    block_4_w_cfg = configer_m['block4_weight']
    block_4_c_cfg = configer_m['block4_conv']

    inputs_4 = [concat_754.get_output(0)]
    weights_4 = get_block_weights(block_4_w_cfg,params)
    network, s_msa_4 = s_msa(network,inputs_4,weights_4,block_4_c_cfg,batch_size,type_id=type_id,name="s_mas4")

    # s_msa_5
    transpose_990 = network.add_shuffle(s_msa_4.get_output(0))
    transpose_990.first_transpose = [0,3,1,2]

    convtranspose_991 = network.add_deconvolution(input=transpose_990.get_output(0),num_output_maps=31,kernel_shape=(2,2),kernel=params["body.0.decoder_layers.1.0.weight"],
        bias=params["body.0.decoder_layers.1.0.bias"])
    convtranspose_991.stride=(2,2)
    convtranspose_991.padding=(0,0)
    convtranspose_991.num_groups=1

    concat_992 = network.add_concatenation([convtranspose_991.get_output(0),transpose_280.get_output(0)])
    concat_992.axis = 1

    block_5_w_cfg = configer_m['block5_weight']
    block_5_c_cfg = configer_m['block5_conv']

    inputs_5 = [concat_992.get_output(0)]
    weights_5 = get_block_weights(block_5_w_cfg,params)
    network, s_msa_5 = s_msa(network,inputs_5,weights_5,block_5_c_cfg,batch_size,type_id=type_id,name="s_mas5")

    # s_msa_6
    transpose_1228 = network.add_shuffle(s_msa_5.get_output(0))
    transpose_1228.first_transpose = [0,3,1,2]

    conv_1229 = network.add_convolution(input=transpose_1228.get_output(0),num_output_maps=31,kernel_shape=(3, 3),kernel=params["body.0.mapping.weight"])
    conv_1229.stride = (1,1)
    conv_1229.padding = (1,1) 
    conv_1229.dilation = (1,1)
    conv_1229.num_groups= 1

    add_1230 = network.add_elementwise(conv_1229.get_output(0),conv_44.get_output(0),trt.ElementWiseOperation.SUM)

    block_6_w_cfg = configer_m['block6_weight']
    block_6_c_cfg = configer_m['block6_conv']

    inputs_6 = [add_1230.get_output(0)]
    weights_6 = get_block_weights(block_6_w_cfg,params)
    network, s_msa_6 = s_msa(network,inputs_6,weights_6,block_6_c_cfg,batch_size,type_id=type_id,name="s_mas6")

    # s_msa_7
    transpose_1466 = network.add_shuffle(s_msa_6.get_output(0))
    transpose_1466.first_transpose = [0,3,1,2]

    block_7_w_cfg = configer_m['block7_weight']
    block_7_c_cfg = configer_m['block7_conv']

    inputs_7 = [transpose_1466.get_output(0)]
    weights_7 = get_block_weights(block_7_w_cfg,params)
    network, s_msa_7 = s_msa(network,inputs_7,weights_7,block_7_c_cfg,batch_size,type_id=type_id,name="s_mas7")

    # s_msa_8
    transpose_1702 = network.add_shuffle(s_msa_7.get_output(0))
    transpose_1702.first_transpose = [0,3,1,2]

    block_8_w_cfg = configer_m['block8_weight']
    block_8_c_cfg = configer_m['block8_conv']

    inputs_8 = [transpose_1702.get_output(0)]
    weights_8 = get_block_weights(block_8_w_cfg,params)
    network, s_msa_8 = s_msa(network,inputs_8,weights_8,block_8_c_cfg,batch_size,type_id=type_id,name="s_mas8")

    # s_msa_9
    transpose_1938 = network.add_shuffle(s_msa_8.get_output(0))
    transpose_1938.first_transpose = [0,3,1,2]

    convtranspose_1939 = network.add_deconvolution(input=transpose_1938.get_output(0),num_output_maps=62,kernel_shape=(2,2),kernel=params["body.1.decoder_layers.0.0.weight"],
        bias=params["body.1.decoder_layers.0.0.bias"])
    convtranspose_1939.stride=(2,2)
    convtranspose_1939.padding=(0,0)
    convtranspose_1939.num_groups=1

    concat_1940 = network.add_concatenation([convtranspose_1939.get_output(0),transpose_1702.get_output(0)])
    concat_1940.axis = 1

    block_9_w_cfg = configer_m['block9_weight']
    block_9_c_cfg = configer_m['block9_conv']

    inputs_9 = [concat_1940.get_output(0)]
    weights_9 = get_block_weights(block_9_w_cfg,params)
    network, s_msa_9 = s_msa(network,inputs_9,weights_9,block_9_c_cfg,batch_size,type_id=type_id,name="s_mas9")

    # s_msa_10
    transpose_2176 = network.add_shuffle(s_msa_9.get_output(0))
    transpose_2176.first_transpose = [0,3,1,2]

    convtranspose_2177 = network.add_deconvolution(input=transpose_2176.get_output(0),num_output_maps=31,kernel_shape=(2,2),kernel=params["body.1.decoder_layers.1.0.weight"],
        bias=params["body.1.decoder_layers.1.0.bias"])
    convtranspose_2177.stride=(2,2)
    convtranspose_2177.padding=(0,0)
    convtranspose_2177.num_groups=1

    concat_2178 = network.add_concatenation([convtranspose_2177.get_output(0),transpose_1466.get_output(0)])
    concat_2178.axis=1

    block_10_w_cfg = configer_m['block10_weight']
    block_10_c_cfg = configer_m['block10_conv']

    inputs_10 = [concat_2178.get_output(0)]
    weights_10 = get_block_weights(block_10_w_cfg,params)
    network, s_msa_10 = s_msa(network,inputs_10,weights_10,block_10_c_cfg,batch_size,type_id=type_id,name="s_mas10")

    #s_msa_11
    transpose_2414 = network.add_shuffle(s_msa_10.get_output(0))
    transpose_2414.first_transpose = [0,3,1,2]

    conv_2415 = network.add_convolution(input=transpose_2414.get_output(0),num_output_maps=31,kernel_shape=(3, 3),kernel=params["body.1.mapping.weight"])
    conv_2415.stride = (1,1)
    conv_2415.padding = (1,1) 
    conv_2415.dilation = (1,1)
    conv_2415.num_groups= 1

    add_2416 = network.add_elementwise(conv_2415.get_output(0),add_1230.get_output(0),trt.ElementWiseOperation.SUM)

    block_11_w_cfg = configer_m['block11_weight']
    block_11_c_cfg = configer_m['block11_conv']

    inputs_11 = [add_2416.get_output(0)]
    weights_11 = get_block_weights(block_11_w_cfg,params)
    network, s_msa_11 = s_msa(network,inputs_11,weights_11,block_11_c_cfg,batch_size,type_id=type_id,name="s_mas11")

    # s_msa_12
    transpose_2652 = network.add_shuffle(s_msa_11.get_output(0))
    transpose_2652.first_transpose = [0,3,1,2]

    block_12_w_cfg = configer_m['block12_weight']
    block_12_c_cfg = configer_m['block12_conv']

    inputs_12 = [transpose_2652.get_output(0)]
    weights_12 = get_block_weights(block_12_w_cfg,params)
    network, s_msa_12 = s_msa(network,inputs_12,weights_12,block_12_c_cfg,batch_size,type_id=type_id,name="s_mas12")

    # s_msa_13
    transpose_2888 = network.add_shuffle(s_msa_12.get_output(0))
    transpose_2888.first_transpose = [0,3,1,2]

    block_13_w_cfg = configer_m['block13_weight']
    block_13_c_cfg = configer_m['block13_conv']

    inputs_13 = [transpose_2888.get_output(0)]
    weights_13 = get_block_weights(block_13_w_cfg,params)
    network, s_msa_13 = s_msa(network,inputs_13,weights_13,block_13_c_cfg,batch_size,type_id=type_id,name="s_mas13")

    # s_msa_14
    transpose_3124 = network.add_shuffle(s_msa_13.get_output(0))
    transpose_3124.first_transpose = [0,3,1,2]

    convtranspose_3125 = network.add_deconvolution(input=transpose_3124.get_output(0),num_output_maps=62,kernel_shape=(2,2),kernel=params["body.2.decoder_layers.0.0.weight"],
        bias=params["body.2.decoder_layers.0.0.bias"])
    convtranspose_3125.stride=(2,2)
    convtranspose_3125.padding=(0,0)
    convtranspose_3125.num_groups=1

    concat_3126 = network.add_concatenation([convtranspose_3125.get_output(0),transpose_2888.get_output(0)])
    concat_3126.axis=1

    block_14_w_cfg = configer_m['block14_weight']
    block_14_c_cfg = configer_m['block14_conv']

    inputs_14 = [concat_3126.get_output(0)]
    weights_14 = get_block_weights(block_14_w_cfg,params)
    network, s_msa_14 = s_msa(network,inputs_14,weights_14,block_14_c_cfg,batch_size,type_id=type_id,name="s_mas14")

    # s_msa_15
    transpose_3362 = network.add_shuffle(s_msa_14.get_output(0))
    transpose_3362.first_transpose = [0,3,1,2]

    convtranspose_3363 = network.add_deconvolution(input=transpose_3362.get_output(0),num_output_maps=31,kernel_shape=(2,2),kernel=params["body.2.decoder_layers.1.0.weight"],
        bias=params["body.2.decoder_layers.1.0.bias"])
    convtranspose_3363.stride=(2,2)
    convtranspose_3363.padding=(0,0)
    convtranspose_3363.num_groups=1

    concat_3364 = network.add_concatenation([convtranspose_3363.get_output(0),transpose_2652.get_output(0)])
    concat_3364.axis=1

    block_15_w_cfg = configer_m['block15_weight']
    block_15_c_cfg = configer_m['block15_conv']

    inputs_15 = [concat_3364.get_output(0)]
    weights_15 = get_block_weights(block_15_w_cfg,params)
    network, s_msa_15 = s_msa(network,inputs_15,weights_15,block_15_c_cfg,batch_size,type_id=type_id,name="s_mas15")

    # last 
    transpose_3600 = network.add_shuffle(s_msa_15.get_output(0))
    transpose_3600.first_transpose = [0,3,1,2]

    conv_3601 = network.add_convolution(input=transpose_3600.get_output(0),num_output_maps=31,kernel_shape=(3, 3),kernel=params["body.2.mapping.weight"])
    conv_3601.stride = (1,1)
    conv_3601.padding = (1,1) 
    conv_3601.dilation = (1,1)
    conv_3601.num_groups= 1

    add_3602 =  network.add_elementwise(conv_3601.get_output(0),add_2416.get_output(0),trt.ElementWiseOperation.SUM)
    conv_3603 = network.add_convolution(input=add_3602.get_output(0),num_output_maps=31,kernel_shape=(3, 3),kernel=params["conv_out.weight"])
    conv_3603.stride = (1,1)
    conv_3603.padding = (1,1) 
    conv_3603.dilation = (1,1)
    conv_3603.num_groups= 1

    add_3604 =  network.add_elementwise(conv_3603.get_output(0),conv_44.get_output(0),trt.ElementWiseOperation.SUM)
    slice_3609 = network.add_slice(input=add_3604.get_output(0),start=trt.Dims4((0,0,0,0)),shape=trt.Dims4((batch_size,31,512,488)),stride=(1,1,1,1))
    slice_3614 = network.add_slice(input=slice_3609.get_output(0),start=trt.Dims4((0,0,0,0)),shape=trt.Dims4((batch_size,31,512,482)),stride=(1,1,1,1))

    #output
    slice_3614.get_output(0).name = "output"
    network.mark_output(slice_3614.get_output(0))


    return network, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorRT API and Plugin for MST++ Model, and get MST++ TRT Engine, support FP32, FP16 and INT8 mode !!!')
    parser.add_argument('--mode', type= str , default='FP32', help='FP32, FP16 or INT8')    
    parser.add_argument('--calibration_table_path', type=str,default='./model/mst_calibration.cache', help="INT8 calibration cache, invalid for FP32 and FP16")
    parser.add_argument('--batch_size',type=int,default=1, help="Batch Size")
    parser.add_argument('--plan_path', type=str,default='./model/mst_plus_plu_b1.plan', help="TRT Engine save path")
    parser.add_argument('--weight_path', type=str,default='./model/mst_plus_plus_weights_b1.npz', help="weight path")

    args = parser.parse_args()

    mode = args.mode
    batch_size =  args.batch_size
    calibration_table_path = args.calibration_table_path
    plan_path = args.plan_path
    weight_path = args.weight_path
    type_id = np.array([0],dtype=np.int32)

    # load weights
    params = np.load(weight_path)

    if mode == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)
        type_id = np.array([1],dtype=np.int32)
        print('FP16 mode enabled')
    if mode == "INT8":
        from calibrator import *
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUDNN))
        config.set_flag(trt.BuilderFlag.INT8)
        # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        calibration_stream = DataLoader(batch_size=batch_size)
        assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
        config.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)

        print('INT8 mode enabled')


    network,config = mst_plus_plus_trt_api(network,config,params,batch_size=batch_size,type_id=type_id)

    plan = builder.build_serialized_network(network, config)
    # engine = runtime.deserialize_cuda_engine(plan)

    #save engine
    with open(plan_path, "wb") as f:
        f.write(plan)
    
    print("TensorRT engine build successful!!!")





    










    























    










