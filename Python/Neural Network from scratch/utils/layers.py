import numpy as np
from typing import List
from .basics import Operation, ParametersOperation, WeightMul, BiasAdd, Flatten
from .convolution import Conv2D_Op
from .activations import Linear
from .dropout import Dropout
from .utility_functions import assert_same_shape

class Layer(object):

    def __init__(self, neurons: int): 
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []


    # This method must be implemented for each Layer
    def _setup_layer(self, input_: np.ndarray) -> None: 
        raise NotImplementedError()


    def _params(self) -> np.ndarray:
        self.params = [] # Empty params

        # Extract the _params from layer's operations with parameters
        for operation in self.operations:
            if issubclass(operation.__class__, ParametersOperation):
                self.params.append(operation.param)


    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        # Set up the first layer
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        # Perform the operations
        for operation in self.operations:
            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output


    def _param_grads(self) -> np.ndarray:
        self.param_grads = [] # Empty param_grads

        # Extract the _param_grads from layer's operations with parameters
        for operation in self.operations:
            if issubclass(operation.__class__, ParametersOperation):
                self.param_grads.append(operation.param_grad)


    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert_same_shape(self.output, output_grad) 

        # Perform operations on gradients
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad
    


class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation = Linear(), conv_in: bool = False, dropout: float = 1.0, weight_init: str = 'standard') -> None:
        super().__init__(neurons)
        self.activation = activation
        self.conv_in = conv_in
        self.dropout = dropout
        self.weight_init = weight_init


    def _setup_layer(self, input_: np.ndarray) -> None:
        # Use the seed of the Neural Network (if present)
        if self.seed:
            np.random.seed(self.seed)

        num_in = input_.shape[1]

        if self.weight_init == 'glorot':
            scale = 2 / (num_in + self.neurons)
        else:
            scale = 1.0

        # Weights
        self.params = [] # Empty params
        self.params.append(np.random.normal(loc = 0, scale = scale, size = (num_in, self.neurons)))

        # Bias
        self.params.append(np.random.normal(loc = 0, scale = scale, size = (1, self.neurons)))

        self.operations = [WeightMul(self.params[0]), BiasAdd(self.params[1]), self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None
    


class Conv2D(Layer):

    def __init__(self, out_channels: int, param_size: int, dropout: int = 1.0, weight_init: str = "normal", activation: Operation = Linear(), flatten: bool = False) -> None:
        super().__init__(out_channels)
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten
        self.dropout = dropout
        self.weight_init = weight_init
        self.out_channels = out_channels

    
    def _setup_layer(self, input_: np.ndarray) -> np.ndarray:
        
        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2/(in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(loc = 0, scale = scale, size = (input_.shape[1], self.out_channels, self.param_size, self.param_size))

        self.params.append(conv_param)

        self.operations = []
        self.operations.append(Conv2D_Op(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(Flatten())

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None