import numpy as np
import sys
from tflite_runtime.interpreter import Interpreter

interpreter=Interpreter('model9-QAT_quantized.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
print(input_details)
print(output_details)
loaded_quantized_inputs_1 = np.load('quantized_inputs_2_1.npy')
loaded_quantized_inputs_2 = np.load('quantized_inputs_2_2.npy')

def dequantize_output(data, scale, zero_point):
    return (data.astype(np.float32) - zero_point) * scale

scale_output = 0.00390625
zero_point_output = -128

for input_1,input_2 in zip(loaded_quantized_inputs_1,loaded_quantized_inputs_2):
    interpreter.set_tensor(input_details[0]['index'], input_1)
    interpreter.set_tensor(input_details[1]['index'], input_2)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    dequantized_output = dequantize_output(output_data, scale_output, zero_point_output)
    print(dequantized_output)
