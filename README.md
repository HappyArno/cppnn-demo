# C++ Neural Network - Demo

A simple C++23 neural network implementation for learning and demonstration purposes, including a simple linear algebra support, linear and non-linear layers, loss functions, and example models for handwritten digit recognition based on the MNIST database.

## Build & Run the Example

1. Execute the following commands in this project directory to compile the utility program for training, testing, and running the handwritten digit recognition example model based on the MNIST database:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

**NOTE:** To simplify the code, model data is allocated on the stack. On Windows, sufficient stack space is automatically reserved during compilation, while on Linux, you need to use `ulimit` to reserve enough stack space.

2. Download the MNIST database and unzip it.

**NOTE:** The resources on [the official website of the MNIST database](https://yann.lecun.com/exdb/mnist/) seem to be no longer available. You can access the mirror resources using the following links:

- [train-images-idx3-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz)
- [train-labels-idx1-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz)
- [t10k-images-idx3-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz)
- [t10k-labels-idx1-ubyte.gz](https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz)

3. Train the model

```bash
./mnist train --model_type MLP --model_path model.bin --epoch_size 5
```

4. Test the model

```bash
./mnist test --model_type MLP --model_path model.bin
```

5. Use [`draw.html`](./draw.html) to draw the digit you want to recognize and save it as a portable graymap format (PGM) file.

6. Recognize the digit

```bash
./mnist run image.pgm --model_type MLP --model_path model.bin
```

The complete help information for this sample program:

```
Usage: ./mnist <command> [options] [operands]
Commands:
    help: Display help information
    train: Train a model
        Options:
        --model_type <string>: Specify the model type (MLP/CNN)
        --model_path <string>: Specify the model path
        --image_path <string>: Specify the training set image file path
        --label_path <string>: Specify the training set label file path
        --epoch_size <integer>: Specify the epoch size
        --batch_size <integer>: Specify the batch size
        --learning_rate <double>: Specify the learning rate
        --weight_decay_rate <double>: Specify the weight decay rate
        --seed <integer>: Specify the seed for generating random numbers
    test: Test a model
        Options:
        --model_type <string>: Specify the model type (MLP/CNN)
        --model_path <string>: Specify the model path
        --image_path <string>: Specify the test set image file path
        --label_path <string>: Specify the test set label file path
    run:
        Operands: The path of the file to be recognized
        Options:
        --model_type <string>: Specify the model type (MLP/CNN)
        --model_path <string>: Specify the model path
```

**NOTE:** You can also use the following command to get this help information:

```bash
./mnist help
```

## Code Structure

- [`utility.hpp`](./utility.hpp): Provides some practical functions.
- [`data_structure.hpp`](./data_structure.hpp): Implements `Vec`, `Array2D`, `Matrix`, `Image`, and `Images`, providing support for data storage and linear algebra operations.
- [`layer.hpp`](./layer.hpp): Implements some linear layers, convolution layers, dropout layers and non-linear layers.
- [`loss.hpp`](./loss.hpp): Implements some loss functions.
- [`mnist_data.hpp`](./mnist_data.hpp): Implements the reading and storage of MNIST database.
- [`mnist_model.hpp`](./mnist_model.hpp): Implements example models of multilayer perceptron (MLP) and convolutional neural network (CNN) for handwritten digit recognition based on the MNIST database.
- [`mnist.cpp`](./mnist.cpp): Implements a utility for training, testing, and running example models.
- [`draw.html`](./draw.html): A simple HTML page for drawing grayscale images and saving them in portable graymap format (PGM).

## License

Copyright (C) 2024 HappyArno

This program is released under the [MIT license](./LICENSE).