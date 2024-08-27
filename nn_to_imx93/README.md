## This branch contains the models, files and scripts that are required to run the model on NPU(IMX93-EthosU65)

1. The "model*-QAT_quantized.tflite" files are Quantization Aware Trained Neural Network models converted to tflite models with int8 quantization. 

2. The "tflite*.h" are files that have the testing data stored in the form of arrays. The data has been extracted from the cuvette csv files and kept in thsi format so that they can be accessed easily. 

3. The "input.py" script prepares the testing data for inference. It takes in data from the "tflite*.h" files and then separates the extracted data to features and labels. Next, it applies the scaling factors to the data so that they can be prepared in int8 format, which is the format our tflite model expects. Before scaling, the features are also normalized with the mean and standard dev values used to normalize the training data, ensuring homogenity across the board. After normalization and scaling, the features are stored in 2 separate files i.e "quantized_inputs_<>_1.npy" and "quantized_inputs_<>_2.npy" as the model takes in 2 inputs, furthermore, the labels are also stored in "labels_<>.npy".

4. The "infer_cuvette*.py" files are inference scripts that loads the tflite model, ethosu delegate (lib to optimise and place the model on NPU), the previously prepared input data and the labels. It gives class probabiltiies and calculates accuracy. 

5. The "cpu_model9.py" is a script that will run the inference with model9-QAT_quantized.tflite on the target device's CPU.

## Instructions on how to run the models

1. Log into the device using ssh. 

    $ ssh root@169.254.165.52
   
   and then CD into the npu directory 
     
    $ cd npu 
    
2. You will be in the /home/root/npu directory now. The first step is to run the input.py file so that we can prepare the inputs for inference using the NN models. 

    $ python3 input.py

   With the inputs now prepared we can run inference.
 
3. If you want to run inference for example, cuvette 9 data, you will use the following command.

    $ python3 infer_cuvette9.py model9-QAT_quantized.tflite ./../../../usr/lib/libethosu_delegate.so

   In case you want to run on any other model.

    $ python3 infer_cuvette*.py model*-QAT_quantized.tflite ./../../../usr/lib/libethosu_delegate.so

   With * representing the different cuvette numbers. 