<h1 align="center"> pyopencl-ml </h1>
<h4 align="center">An AMD APU compatible machine learning library with pyopencl.</h4>

<p align="center">
    <img src="https://img.shields.io/github/languages/top/dasanish/pyopencl-ml"> •
    <img src="https://img.shields.io/github/last-commit/dasanish/pyopencl-ml"> 
</p>

<h3 align="center"> Content </h3>
<p align="center">
    <a href="#about"> About </a> •
    <a href="#installation"> Installation </a> •
    <a href="#usage"> Usage </a> •
    <a href="#contributing"> Contributing </a> •
    <a href="#authors"> Authors </a> •
    <a href="#license"> License </a>
</p>

## About
Most machine learning frameworks in python don't have GPU support for AMD-GPUs 
and thus no convenient way to run machine learning algorithms on windows with AMD APU.
This project was created to provide a starting point and is very much in it's 
infancy at the moment and should improve with time.

## Installation
* Installing OpenCL
    - recently the AMD sdk for opencl was discontinue but 
    [OCL-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)
    has worked so far.
    - download & extract the zip 
    - set / append Environment Variables `LIB` to `lib/x86_64` && `INCLUDE` to `include`
* Installing PyOpenCL
    - install in cmd using: `pip install pyopencl`. If it doesn't work then try
    ```
    pip3 install --global-option=build_ext --global-option="-I[ADJUST_PATH]\OCL_SDK_Light\include" --global-option="-L[ADJUST_PATH]\OCL_SDK_Light\lib\x86_64" pyopencl
    ```

## Usage
1. Creating a NN object
    ```buildoutcfg
    from clobject import *
    from layer import Layer, Output
    from neuralnet import NeuralNet

    nn_object = NeuralNet(
       Layer(INPUT_SIZE, ACTIVATION_TYPE),
       Layer(HIDDEN_LAYER_SIZE, ACTIVATION_TYPE),
       .
       .
       Output(OUTPUT_SIZE)
    )
    ```

2. both input and output vectors must be of type `numpy.ndarray` with `dtype('float32')`
    * if each input-output vector is spliced then use \
    `nn_object.train(dataset, batch_size, num_epochs, print_every)`
    * if input and output vectors are seperate then use \
    `nn_object.fit(train_X, train_y, batch_size, num_epochs, print_every)`
    
3. Save the NN: #TODO 
    

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Author(s)
- Anish Das (ad945@cam.ac.uk)

## License 
[![License: MIT](https://img.shields.io/github/license/dasanish/pyopencl-ml)](https://choosealicense.com/licenses/mit/)