import numpy as np
from .basics import Operation


class Linear(Operation):

    def __init__(self) -> None:        
        super().__init__()


    def _output(self, inference: bool) -> np.ndarray:
        return self.input_


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad




class Sigmoid(Operation):

    def __init__(self) -> None:
        super().__init__()


    def _output(self, inference: bool) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad



class Tanh(Operation):

    def __init__(self) -> None:
        super().__init__()


    def _output(self, inference: bool) -> np.ndarray:
        return np.tanh(self.input_)


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray: 
        return output_grad * (1 - self.output * self.output)



class ReLU(Operation):

    def __init__(self) -> None:
        super().__init__()


    def _output(self, inference: bool) -> np.ndarray:
        return np.clip(self.input_, 0, None)


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        non_neg = (self.output >= 0)
        return output_grad * non_neg