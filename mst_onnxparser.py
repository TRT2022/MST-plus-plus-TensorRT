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
ONNXParser MST++ TemsorRT INT8量化 calibrator  for  PTQ！
'''


import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import argparse
# add verbose

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # ** engine可视化 **

# create tensorrt-engine
# fixed and dynamic
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",\
              mode="FP32", calibration_stream=None, calibration_table_path="", save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1) as network,\
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:
            
            # parse onnx model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
                assert network.num_layers > 0, 'Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible '
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))        
            
            # build trt engine
            builder.max_batch_size = max_batch_size
            config.max_workspace_size = 1 << 30 # 1GB

            if mode == "FP16":
                config.set_flag(trt.BuilderFlag.FP16)
                print('FP16 mode enabled')
           
            if mode == "INT8":
                config.set_flag(trt.BuilderFlag.INT8)
                assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                config.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)
                print('Int8 mode enabled')

            # Build engine and do int8 calibration.
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                print('Failed to create the engine')
                return None   
            print("Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(plan)
            return engine
        
    # if os.path.exists(engine_file_path):
    #     # If a serialized engine exists, load it instead of building a new one.
    #     print("Reading engine from file {}".format(engine_file_path))
    #     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         return runtime.deserialize_cuda_engine(f.read())
    # else:
    return build_engine(max_batch_size, save_engine)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorRT API and Plugin for MST++ Model, and get MST++ TRT Engine, support FP32, FP16 and INT8 mode !!!')
    parser.add_argument('--mode', type= str , default='FP32', help='FP32, FP16 or INT8')    
    parser.add_argument('--calibration_table_path', type=str,default='./model/mst_calibration_onnxparser_b1.cache', help="INT8 calibration cache, invalid for FP32 and FP16")
    parser.add_argument('--batch_size',type=int,default=1, help="Batch Size")
    parser.add_argument('--plan_path', type=str,default='./model/mst_plus_plus_onnxparser_b1_int8.plan', help="TRT Engine save path")
    parser.add_argument('--onnx_path', type=str,default='./model/mst_plus_plus_b1.onnx', help="ONNX path")

    args = parser.parse_args()

    mode = args.mode
    batch_size =  args.batch_size
    calibration_table_path = args.calibration_table_path
    plan_path = args.plan_path
    onnx_path = args.onnx_path

    if mode == "FP32":
        get_engine(max_batch_size=batch_size, onnx_file_path=onnx_path, engine_file_path=plan_path,\
              mode="FP32",save_engine=True)
    if mode == "FP16":
        get_engine(max_batch_size=batch_size, onnx_file_path=onnx_path, engine_file_path=plan_path,\
              mode="FP16",save_engine=True)
    if mode == "INT8":
        from calibrator import *
        calibration_stream = DataLoader(batch_size=batch_size)
        assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
        print('INT8 mode enabled')
        get_engine(max_batch_size=batch_size, onnx_file_path=onnx_path, engine_file_path=plan_path,\
              mode="INT8", calibration_stream=calibration_stream, calibration_table_path=calibration_table_path, save_engine=True)


