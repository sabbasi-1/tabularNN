import numpy as np
import time
import random
import tflite_runtime.interpreter as tflite
import re
import sys
import os

def parse_header_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        matches = re.findall(r'\{(.*?)\}', data, re.DOTALL)

    parsed_data = []
    for match in matches:
        cleaned = match.replace('{', '').replace('}', '').strip()
        values = list(map(int, re.findall(r'\d+', cleaned)))
        parsed_data.append(values)
    return parsed_data

def separate_features_and_labels(parsed_data):
    features = [entry[:-1] for entry in parsed_data]
    labels = [entry[-1] for entry in parsed_data]
    return features, labels

def quantize_input(data, scale, zero_point):
    normalized_data = (data / scale) + zero_point
    quantized_data = np.clip(normalized_data, -128, 127).astype(np.int8)
    return quantized_data
def dequantize_output(data, scale, zero_point):
    return (data.astype(np.float32) - zero_point) * scale

def pick_random_samples(features, labels, num_samples, seed=None):
    if seed is not None:
        random.seed(seed)
    combined = list(zip(features, labels))
    random_samples = random.sample(combined, num_samples)
    sampled_features, sampled_labels = zip(*random_samples)
    return list(sampled_features), list(sampled_labels)

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


    scale_input_1 = 21384.0390625
    zero_point_input_1 = -128
    scale_input_2 = 21384.0390625
    zero_point_input_2 = -128
    scale_output = 0.00390625
    zero_point_output = -128


    header_file_path = 'tflite29.h'
    parsed_data = parse_header_file(header_file_path)
    features, labels = separate_features_and_labels(parsed_data)


    num_samples = 1000
    seed = 42
    features, labels = pick_random_samples(features, labels, num_samples, seed)

    labels = np.array(labels)
    features = np.array(features)
    print(features.shape)

    correct_predictions = 0
    total_inference_time = 0

    for i in range(len(features)):
        start_time = time.time()
        input = features[i].astype(np.float32)
        input = np.expand_dims(input, axis=0)

        quantized_input_1 = quantize_input(input, scale_input_1, zero_point_input_1)
        quantized_input_2 = quantize_input(input, scale_input_2, zero_point_input_2)
        interpreter.set_tensor(input_details[0]['index'], quantized_input_1)
        interpreter.set_tensor(input_details[1]['index'], quantized_input_2)

        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1e3  # Convert to milliseconds
        total_inference_time += inference_time
        quantized_output = interpreter.get_tensor(output_details[0]['index'])
        probability = dequantize_output(quantized_output, scale_output, zero_point_output)
        prob = np.squeeze(probability)
        predicted_label = 1 if prob > 0.5 else 0

        if predicted_label == labels[i]:
            correct_predictions += 1

   
    accuracy = (correct_predictions / len(labels)) * 100

    print(f'Accuracy: {accuracy:.2f}%')
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