## Steps to Run the Notebook and Train Neural Network Models for Different Cuvettes

1. Before we start running the cells in the notebook, we need to setup an environment and a kernel where the required packages are already installed.

2. The first step is to setup a conda environment. In case you don't have conda installed in your system, use the following comamnds to do so.

       $ mkdir -p ~/miniconda3
       $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
       $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
       $ rm -rf ~/miniconda3/miniconda.sh
       $ ~/miniconda3/bin/conda init bash
       $ ~/miniconda3/bin/conda init zsh

4. Now that you conda setup, create a new conda environment using.

       $ conda create -n <env-name> python==3.10.13

5. Next, we need to install the packages that will be required in the notebook. First we will activate the new environment and then install the packages using the "requirements.txt" file present in the branch.

       $ conda activate <env-name>
       $ python -m pip install -r requirements.txt

6. When the packages are all done, installed and ready, now we need to create a kernel that can be used within the jupyter notebook. For that, we will install ipykernel package. We will also install jupyter in this step.

       $ conda install jupyter
       $ conda install ipykernel

7. Now, to create a kernel of our conda environment. Use the ipykernel pacakge in the following way.

       $ python -m ipkernel --user --name=<env-name> --display-name "<the kernel name to show in notebook>"

8. With this you are all set to work on the notebook without any problems. Open up a server and use the notebook over there.

## Information Regarding the Notebook

1. The notebook as suggested in the start deals with training NN models for different cuvettes. Each model is trained separately as there is difference in the cuvette data.

2. Models for cuvette 45 and 47 have the same architectures and similarly, models for cuvette 0 and 37 have similar architectures. Other models are different to every other model.

3. For each cuvette, the work flow is as follows.

       a. The relative training csv file (named as data-train) is loaded and processed to separate the features and labels.
       b. Next, the features are scaled using the Scikit-Learn Standard Scaler.
       c. In the third step, the model architecture is defined using tensorflow and the model is set up for Quantization Aware Training which is important for tflite conversion without a significant drop in performance.
       d. Once the relative hyperparameters are defined, the model is set to train. Each model in the notebook is being trained for 15 epochs, but that can be  changed for longer training.
       e. Once, the model is trained, the model is evaluated on unseen data or test data. The test data is loaded from the csv files named data.csv in each cuvette folder.
       f. After evaluation on unseen performance, the steps for converting the model to tflite and quantizing it in the process starts.
       g. A representative dataset is generated so that the inputs and the model can be quantized in a much more coherrent way. The represenatative dataset is also taken from the training csv files.
       h. With the representative dataset defined, the model is quantized and converted to tflite format.
       i. In the next step, it is verified that the model has been quantized to int8 format or not. It is seen by printing out the input data type and output  data type of the tflite model.
       j. Once, the int8 quantization is verified the next step is to run inference with the tflite models to check their performance and see if there is a significant dropout in performance or not.
       k. The input and ouput tensor details are printed and their scales are noted down as they are an important cog in preprocessing the unseen data before sending it to the tflite model.
       l. Next, the unseen data is loaded from the csv file, scaled with the same values as we did our training data and then converted to int8 format using the scales noted down in the previous step.
       m. The tflite interpreter is instantiated and then the input tensor are allocated to it. It runs inference, gives an output which is also scaled back up to its real value and the accuracy is calculated and printed out.
       n. All of this process is repeated for each cuvette.

4. After the training, conversion and verification of the performance of the cuvette models. There are 2 more important steps to take.

5. One is creating a header file of the inference data present in a format inside the file that is representative of the csv file.

6. And the last step is to print out the mean and std dev values for each cuvette data so the data can be scaled in the inference scripts being run on the target devices. 
   