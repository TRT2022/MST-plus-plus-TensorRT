
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
MST++ Pytorch模型转ONNX, 同时支持Static, Dynamic shape和onnx-simplifier

'''

import os
import argparse

import architecture
import torchsummary
import torch
import torch.onnx
import onnx
from onnxsim import simplify

import warnings
warnings.filterwarnings("ignore")

def pth_to_onnx(checkpoint, onnx_path, input_names=['input'], output_names=['output'],batch_size=1,dynamic=False,simplifier=False):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    with torch.no_grad():
        model =  architecture.MST_Plus_Plus() # 导入模型
        check_point = torch.load(checkpoint)
        model.load_state_dict(check_point['state_dict'])  # 初始化权重
        model.eval()
        model.to("cpu")
        torchsummary.summary(model, (3, 512, 482), device='cpu')

        if dynamic:
            input = torch.randn(1, 3, 512, 482, device='cpu')
            device = torch.device('cpu')
            dynamic_axes = {input_names[0]:{0:"batch_size"},output_names[0]:{0:"batch_size"}}
            torch.onnx.export(model, input, onnx_path, verbose=False, input_names=input_names,
                            output_names=output_names, opset_version=13,
                            dynamic_axes=dynamic_axes)  
        else:
            input = torch.randn(batch_size, 3, 512, 482, device='cpu')
            device = torch.device('cpu')
            torch.onnx.export(model, input, onnx_path, verbose=False, input_names=input_names,
                    output_names=output_names, opset_version=13) 

        if simplifier:
            #simplifier onnx
            model = onnx.load(onnx_path)
            # simplifying dynamic model
            simplified_model, check = simplify(model,
                                            input_shapes={input_names[0]: [batch_size, 3, 512, 482]},
                                            dynamic_input_shape=dynamic)

            onnx.save(simplified_model,onnx_path)

    
    
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MST++ Pytorch Model to ONNX Model')
    parser.add_argument('--checkpoint', type= str , default='./model/mst_plus_plus.pth', help='MST++ Pytorch Model Saved Dir')    
    parser.add_argument('--dynamic', action="store_true", help='Dynamic ONNX Model')
    parser.add_argument('--onnx_path', type=str,default='./model/mst_plus_plus.onnx', help="ONNX saved path")
    parser.add_argument('--simplifier',action="store_true", help='onnx-simplifier ONNX Model')
    parser.add_argument('--batch_size',type=int,default=1, help="Batch Size")

    args = parser.parse_args()

    checkpoint = args.checkpoint
    onnx_path =  args.onnx_path
    dynamic = args.dynamic
    simplifier = args.simplifier
    batch_size = args.batch_size
    pth_to_onnx(checkpoint, onnx_path,batch_size=batch_size,dynamic=dynamic,simplifier=simplifier)
