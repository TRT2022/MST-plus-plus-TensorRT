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
获取模型权重用来在TensorRT API中加载

'''

import onnx
from onnx import numpy_helper
import numpy as np
import argparse


def get_weights(model_path,weight_path):
    model = onnx.load(model_path)

    torchParam = {}
    for t in model.graph.initializer:
        # print(t.name)
        # print(numpy_helper.to_array(t).shape)
        # print(numpy_helper.to_array(t))
        torchParam[t.name] = numpy_helper.to_array(t)


    np.savez(weight_path,**torchParam)
    # params = np.load("./yolact_weights.npz")
    print("model weights save successful!!!")

    return

if __name__ == "__main__":
    # model_path = "./model/mst_plus_plus.onnx"
    # weight_path = "./model/mst_plus_plus_weights.npz"
    parser = argparse.ArgumentParser(description='MST++ get model weight')
    parser.add_argument('--model_path', type= str , default='./model/mst_plus_plus_b1.onnx', help='MST++ model path')    
    parser.add_argument('--weight_path', type=str,default='./model/mst_plus_plus_weights_b1.npz', help="weight saved path")

    args = parser.parse_args()

    model_path = args.model_path
    weight_path =  args.weight_path
    get_weights(model_path,weight_path)



