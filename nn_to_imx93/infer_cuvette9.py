import sys
import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite

def dequantize_output(data, scale, zero_point):
    return (data.astype(np.float32) - zero_point) * scale

def run_model(model_file, ext_delegate_file=None):
    if ext_delegate_file:
        print(f'Loading external delegate from {ext_delegate_file}')
        ext_delegate = [tflite.load_delegate(ext_delegate_file)]
    else:
        ext_delegate = None

    interpreter = tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=ext_delegate,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input1 = np.load('quantized_inputs_2_1.npy')
    input2 = np.load('quantized_inputs_2_2.npy')
    labels = np.load('labels_2.npy')

    correct_predictions = 0
    scale_output = 0.00390625
    zero_point_output = -128
    total_inference_time = 0

    for input_1, input_2, label in zip(input1, input2, labels):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_1)
        interpreter.set_tensor(input_details[1]['index'], input_2)
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1e3  # Convert to milliseconds
        total_inference_time += inference_time
        output_data = interpreter.get_tensor(output_details[0]['index'])
        dequantized_output = dequantize_output(output_data, scale_output, zero_point_output)
        prob = np.squeeze(dequantized_output)
        predicted_label = 1 if prob > 0.5 else 0

        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(labels) *100

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Total inference time: {total_inference_time:.2f} ms')
    print(f'Average inference time per sample: {total_inference_time / len(labels):.2f} ms')

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python run_model.py <model_file> [<ext_delegate_file>]")
        sys.exit(1)
    
    model_file = sys.argv[1]
    ext_delegate_file = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        sys.exit(1)

    if ext_delegate_file and not os.path.exists(ext_delegate_file):
        print(f"External delegate file not found: {ext_delegate_file}")
        sys.exit(1)

    run_model(model_file, ext_delegate_file)