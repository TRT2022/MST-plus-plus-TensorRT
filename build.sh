# 1.Pytorch模型转ONNX

python3 torch2onnx --batch_size=1 --onnx_path=./model/mst_plus_plus_b1.onnx --simplifier
python3 torch2onnx --batch_size=2 --onnx_path=./model/mst_plus_plus_b2.onnx --simplifier
python3 torch2onnx --batch_size=4 --onnx_path=./model/mst_plus_plus_b4.onnx --simplifier
python3 torch2onnx --batch_size=8 --onnx_path=./model/mst_plus_plus_b8.onnx --simplifier
python3 torch2onnx --batch_size=16 --onnx_path=./model/mst_plus_plus_b16.onnx --simplifier


# 2.TensorRT ONNXParser模型序列化
#FP32
trtexec --onnx=./model/mst_plus_plus_b1.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b1.plan --workspace=3000 --verbose
trtexec --onnx=./model/mst_plus_plus_b2.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b2.plan --workspace=3000 --verbose
trtexec --onnx=./model/mst_plus_plus_b4.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b4.plan --workspace=3000 --verbose
trtexec --onnx=./model/mst_plus_plus_b8.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b8.plan --workspace=3000 --verbose
trtexec --onnx=./model/mst_plus_plus_b16.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b16.plan --workspace=3000 --verbose

#FP16
trtexec --onnx=./model/mst_plus_plus_b1.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b1_fp16.plan --workspace=3000 --verbose --fp16
trtexec --onnx=./model/mst_plus_plus_b2.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b2_fp16.plan --workspace=3000 --verbose --fp16
trtexec --onnx=./model/mst_plus_plus_b4.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b4_fp16.plan --workspace=3000 --verbose --fp16
trtexec --onnx=./model/mst_plus_plus_b8.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b8_fp16.plan --workspace=3000 --verbose --fp16
trtexec --onnx=./model/mst_plus_plus_b16.onnx --saveEngine=./model/mst_plus_plus_onnxparser_b16_fp16.plan --workspace=3000 --verbose --fp16

# 3.使用Nsight Profiling ONNXParser
#nsys profile -o mst_onnx trtexec --loadEngine=./model/mst_plus_plus_onnxparser_b1.plan --warmUp=0 --duration=0 --iterations=50 --plugins=./LayerNormPlugin.so

# plugin
cd LayerNormPlugin
make
mv LayerNormPlugin.so ../
cd ..

cd NormalizePlugin
make
mv NormalizePlugin.so ../
cd ..

# 4.TensorRT API 模型序列化
# FP32
python3 mst_trt_api.py --batch_size=1 --mode=FP32 --plan_path=./model/mst_plus_plus_b1.plan --weight_path=./model/mst_plus_plus_weights_b1.npz
python3 mst_trt_api.py --batch_size=2 --mode=FP32 --plan_path=./model/mst_plus_plus_b2.plan --weight_path=./model/mst_plus_plus_weights_b2.npz
python3 mst_trt_api.py --batch_size=4 --mode=FP32 --plan_path=./model/mst_plus_plus_b4.plan --weight_path=./model/mst_plus_plus_weights_b4.npz
python3 mst_trt_api.py --batch_size=8 --mode=FP32 --plan_path=./model/mst_plus_plus_b8.plan --weight_path=./model/mst_plus_plus_weights_b8.npz
python3 mst_trt_api.py --batch_size=16 --mode=FP32 --plan_path=./model/mst_plus_plus_b16.plan --weight_path=./model/mst_plus_plus_weights_b16.npz

# FP6
python3 mst_trt_api.py --batch_size=1 --mode=FP16 --plan_path=./model/mst_plus_plus_b1_fp16.plan --weight_path=./model/mst_plus_plus_weights_b1.npz
python3 mst_trt_api.py --batch_size=2 --mode=FP16 --plan_path=./model/mst_plus_plus_b2_fp16.plan --weight_path=./model/mst_plus_plus_weights_b2.npz
python3 mst_trt_api.py --batch_size=4 --mode=FP16 --plan_path=./model/mst_plus_plus_b4_fp16.plan --weight_path=./model/mst_plus_plus_weights_b4.npz
python3 mst_trt_api.py --batch_size=8 --mode=FP16 --plan_path=./model/mst_plus_plus_b8_fp16.plan --weight_path=./model/mst_plus_plus_weights_b8.npz
python3 mst_trt_api.py --batch_size=16 --mode=FP16 --plan_path=./model/mst_plus_plus_b16_fp16.plan --weight_path=./model/mst_plus_plus_weights_b16.npz


#5.使用Nsight Profiling TensorRT API模型
#nsys profile -o mst_trt_api trtexec --loadEngine=./model/mst_plus_plus_b1.plan --warmUp=0 --duration=0 --iterations=50 --plugins=./LayerNormPlugin.so --Plugins=NormalizePlugin.so

# 6.benchmark的计算包括：Pytorch,onnxruntime, TensorRT ONNXParser, TensorRT API的延迟，吞吐量，加速比等
python3 performance_latency.py 

# 7.精度的计算包括： Pytorch,onnxruntime, TensorRT ONNXParser, TensorRT API的绝对和相对误差的均值最大值，中位数
python3 performance_accurcay.py 

# 8.TensorRT API方式的INT8量化
# python3 mst_trt_api.py --batch_size=1 --mode=INT8 --calibration_table_path=./model/mst_calibration_b1.cache --plan_path=./model/mst_plus_plus_b1_int8.plan --weight_path=./model/mst_plus_plus_weights_b1.npz
# python3 mst_trt_api.py --batch_size=2 --mode=INT8 --calibration_table_path=./model/mst_calibration_b2.cache --plan_path=./model/mst_plus_plus_b2_int8.plan --weight_path=./model/mst_plus_plus_weights_b2.npz
# python3 mst_trt_api.py --batch_size=4 --mode=INT8 --calibration_table_path=./model/mst_calibration_b4.cache --plan_path=./model/mst_plus_plus_b4_int8.plan --weight_path=./model/mst_plus_plus_weights_b4.npz
# python3 mst_trt_api.py --batch_size=8 --mode=INT8 --calibration_table_path=./model/mst_calibration_b8.cache --plan_path=./model/mst_plus_plus_b8_int8.plan --weight_path=./model/mst_plus_plus_weights_b8.npz

# 9.TensorRT ONNXParser方式的INT8量化
python3 mst_onnxparser.py --batch_size=1 --mode=INT8 --calibration_table_path=./model/mst_calibration_onnxparser_b1.cache  --onnx_path=./model/mst_plus_plus_b1.onnx --plan_path=./model/mst_plus_plus_onnxparser_b1_int8.plan
python3 mst_onnxparser.py --batch_size=2 --mode=INT8 --calibration_table_path=./model/mst_calibration_onnxparser_b2.cache  --onnx_path=./model/mst_plus_plus_b2.onnx --plan_path=./model/mst_plus_plus_onnxparser_b2_int8.plan
python3 mst_onnxparser.py --batch_size=4 --mode=INT8 --calibration_table_path=./model/mst_calibration_onnxparser_b4.cache  --onnx_path=./model/mst_plus_plus_b4.onnx --plan_path=./model/mst_plus_plus_onnxparser_b4_int8.plan
python3 mst_onnxparser.py --batch_size=8 --mode=INT8 --calibration_table_path=./model/mst_calibration_onnxparser_b8.cache  --onnx_path=./model/mst_plus_plus_b8.onnx --plan_path=./model/mst_plus_plus_onnxparser_b8_int8.plan


# 10.INT8模型的benchmark和精度的计算
python3 performance_int8.py 
