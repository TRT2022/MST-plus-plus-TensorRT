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
MST++ TemsorRT INT8量化 calibrator  for  PTQ！
'''

import os
import glob
import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
import logging
logger = logging.getLogger(__name__)

# calibrator

'''
IInt8EntropyCalibrator2
    Entropy calibration chooses the tensor’s scale factor to optimize the quantized tensor’s information-theoretic content, 
    and usually suppresses outliers in the distribution. This is the current and recommended entropy calibrator and is required for DLA. 
    Calibration happens before Layer fusion by default. It is recommended for CNN-based networks.
IInt8MinMaxCalibrator
    This calibrator uses the entire range of the activation distribution to determine the scale factor. 
    It seems to work better for NLP tasks. Calibration happens before Layer fusion by default. 
    This is recommended for networks such as NVIDIA BERT (an optimized version of Google's official implementation).
IInt8EntropyCalibrator
    This is the original entropy calibrator. It is less complicated to use than the LegacyCalibrator and typically produces better results. 
    Calibration happens after Layer fusion by default.
IInt8LegacyCalibrator
    This calibrator is for compatibility with TensorRT 2.0 EA. 
    This calibrator requires user parameterization and is provided as a fallback option if the other calibrators yield poor results. 
    Calibration happens after Layer fusion by default. 
    You can customize this calibrator to implement percentile max, 
    for example, 99.99% percentile max is observed to have best accuracy for NVIDIA BERT.

PTQ  not QAT（QDQ)
'''


height=482
width=512
CALIB_IMG_DIR = './data'

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_fp32 = np.array(img,dtype=np.float32)
    img_fp32 = (img_fp32 - img_fp32.min()) / (img_fp32.max() - img_fp32.min())

    img_1 = img_fp32.transpose([2,1,0]).reshape(1,3,512,482)  # 3x512x482
    return img_1


class DataLoader:
    def __init__(self,batch_size):
        self.index = 0
        self.batch_size = batch_size
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        self.length = len(self.img_list) // self.batch_size

        print('found all {} images to calib.'.format(len(self.img_list)))
        assert len(self.img_list) >= self.batch_size * self.length, '{} must contains more than '.format(CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
        self.calibration_data = {"input":np.zeros((self.batch_size,3,width,height), dtype=np.float32)}

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                self.calibration_data["input"][i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data["input"], dtype=np.float32)
        else:
            return np.array([])
            
    def __len__(self):
        return self.length



class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)       
        self.stream = stream
        self.d_input_0 = cuda.mem_alloc(self.stream.calibration_data["input"].nbytes)

        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self,names=["input"]):
        batch_img = self.stream.next_batch()
        if not batch_img.size:  
            logger.info("----------------------------------------------------------------->>>>>>")
            logger.info(names)
            return None

        cuda.memcpy_htod(self.d_input_0, batch_img)

        return [int(self.d_input_0)]
        
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)
