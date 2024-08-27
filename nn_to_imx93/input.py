import numpy as np
import random
import re


params = {
    1: {
        'scale1': 0.49051696,
        'zero_point1': -128,
        'scale2': 0.49051696,
        'zero_point2': -128,
        'mean': [5960.01795959,  5854.02377151,  5899.94796707,  6183.13921177,
                6136.55088551,  5288.41606386,  6315.12434522,  5678.19980045,
                7172.70800698, 14510.63716638],
        'stddev': [ 34117.78624936,  30330.56180343,  32265.7709535,   36122.43592408,
                    35673.62541836, 34315.2584925,   36506.19218768,  45582.91716754,
                    86896.89317717, 202895.06966576]
    },
    2: {
        'scale1': 0.0277759,
        'zero_point1': -116,
        'scale2':0.0277759,
        'zero_point2': -116,
        'mean': [263005.00630163, 107228.73537427, 182464.92101678, 305580.336966,
                 315750.93708592, 316616.04445201, 321487.12511283, 335841.54783275,
                 234127.44907605, 311829.21977348],
        'stddev': [870806.30216227, 325278.70199715, 553725.92898619, 1083730.05,
                   1121533.38050594, 1125740.11780339, 1141739.68742835, 1194981.44422409,
                   712991.48054722, 1057456.74927629]
    },
    3: {
        'scale1': 0.06492095,
        'zero_point1': -123,
        'scale2' : 3.3731067,
        'zero_point2':-128, 
        'mean': [214696.88924557, 23818.43831744,  57672.67536614, 106311.83428134,
                113646.35336342, 114540.78697207, 119389.56726839, 129504.56879257,
                97153.15690565, 295715.22246253],
        'stddev': [ 730746.43161753,   72752.5896946,   176497.21949464,  350258.21619293,
                    375907.8088742,   379821.57705414, 396324.98443289,  432989.20608965,
                    316899.23971688, 1042513.73119661]
    },
    4: {
        'scale1': 0.5029321,
        'zero_point1': -127,
        'scale2': 0.5029321,
        'zero_point2': -127,
        'mean': [22279.53101336,  5607.05090615,  5878.98712669,  6744.9753765,
                6786.96835702,  5984.29847698,  7482.49852804,  8019.32012252,
                10552.66953118, 49246.47540203],
        'stddev': [ 75072.37609014,  16830.38911429,  17649.25719835,  20309.27049679,
                    20450.9778758,   18061.54586877, 43302.22244195,  84829.67017441,
                    148278.11190376, 311266.189395  ]
    },
    5: {
        'scale1': 0.01612353,
        'zero_point1': -107,
        'scale2': 0.0614062,
        'zero_point2':-123,
        'mean': [415880.3316878,  312708.3256958,  398829.50905609, 369891.40257894,
                374843.69781258, 375431.95330666, 378360.12639374, 383881.86218402,
                443516.04716146, 449311.49115669],
        'stddev': [1378727.49115844,  948131.64974938, 1210089.90674477, 1228356.01247843,
                  1245925.41936408, 1248701.90572005, 1257861.06935382, 1276336.69423946,
                    1344655.61010865, 1501521.77770609]
    },
    6: {
        'scale1': 0.01622357,
        'zero_point1': -106,
        'scale2': 0.06672344,
        'zero_point2': -123,
        'mean': [608849.04579647, 636529.87719366, 633594.34718126, 595496.84609138,
                591990.60721887, 590820.35166047, 593228.33780827, 594909.16961576,
                644724.65739625, 600410.42277642],
        'stddev': [1957389.16032315, 1985083.97026398, 1982686.45137855, 1929087.66294351,
                    1924312.08642231, 1925582.60693328, 1929780.27595846, 1935818.06190744,
                    2010645.3481033,  1955323.31671569]
    },
}

header_file_paths = [
    'Files/tflite0.h', 'Files/tflite9.h', 'Files/tflite22.h',
    'Files/tflite37.h', 'Files/tflite45.h', 'Files/tflite47.h'
]

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


for i in range(6):
    parsed_data = parse_header_file(header_file_paths[i])
    features, labels = separate_features_and_labels(parsed_data)
    features, labels = pick_random_samples(features, labels, 2000, seed=42)

    mean = np.array(params[i + 1]['mean'])
    stddev = np.array(params[i + 1]['stddev'])
    scale1 = params[i + 1]['scale1']
    zero_point1 = params[i + 1]['zero_point1']
    scale2 = params[i + 1]['scale2']
    zero_point2 = params[i + 1]['zero_point2']
    labels = np.array(labels)

    features = [item for item in features if len(item) == 10]
    features = np.array(features)
    features = (features - mean) / stddev

    quantized_inputs1 = []
    quantized_inputs2=[]
    for feature in features:
        input_data = np.expand_dims(feature.astype(np.float32), axis=0)
        quantized_input1 = quantize_input(input_data, scale1, zero_point1)
        quantized_input2 = quantize_input(input_data, scale2, zero_point2)
        quantized_inputs1.append(quantized_input1)
        quantized_inputs2.append(quantized_input2)

    np.save(f'quantized_inputs_{i + 1}_1.npy', quantized_inputs1)
    np.save(f'quantized_inputs_{i + 1}_2.npy', quantized_inputs2)
    np.save(f'labels_{i + 1}.npy', labels)
