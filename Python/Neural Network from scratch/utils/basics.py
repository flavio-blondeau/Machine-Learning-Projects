import numpy as np
from .utility_functions import assert_same_shape


class Operation(object):

    def __init__(self):
        pass
  
    # This method must be defined for each Operation
    def _output(self, inference: bool) -> np.ndarray:
        raise NotImplementedError()


    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        self.input_ = input_
        self.output = self._output(inference)

        return self.output

    # This method must be defined for each Operation
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray: 
        raise NotImplementedError()


    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        
        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad



class ParametersOperation(Operation):

    def __init__(self, param: np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param

    # This method must be defined for each ParameterOperation
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray: 
        raise NotImplementedError()


    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad



class WeightMul(ParametersOperation):

    def __init__(self, W: np.ndarray):
        super().__init__(W)


    def _output(self, inference: bool) -> np.ndarray:
        return np.matmul(self.input_, self.param) 


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray: 
        return np.matmul(output_grad, np.transpose(self.param, (1,0)))


    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray: 
        return np.matmul(np.transpose(self.input_, (1,0)), output_grad)



class BiasAdd(ParametersOperation):

    def __init__(self, b: np.ndarray):
        super().__init__(b)


    def _output(self, inference: bool) -> np.ndarray: 
        return self.input_ + self.param


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray: 
        return np.ones_like(self.input_) * output_grad


    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray: 
        output_grad_reshaped = np.sum(output_grad, axis=0).reshape(1, -1)
        param_grad = np.ones_like(self.param)
        return param_grad * output_grad_reshaped



class Flatten(Operation):

    def __init__(self):
        super().__init__()


    def _output(self, inference: bool = False) -> np.ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.reshape(self.input_.shape)