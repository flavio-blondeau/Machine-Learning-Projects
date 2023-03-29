import numpy as np
from .utility_functions import assert_same_shape, softmax, normalize, unnormalize


class Loss(object):

    def __init__(self) -> None:
        pass


    # This method must be defined for each Loss
    def _output(self) -> float: 
        raise NotImplementedError()


    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        assert_same_shape(prediction, target)

        self.prediction = prediction 
        self.target = target

        self.output = self._output()

        return self.output


    # This method must be defined for each Loss
    def _input_grad(self) -> np.ndarray:
        raise NotImplementedError()


    def backward(self) -> np.ndarray:
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad



class MeanSquaredError(Loss):

    def __init__(self, normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize


    def _output(self) -> float:

        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(axis = 1, keepdims = True)

        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

        return loss


    def _input_grad(self) -> np.ndarray:

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
    


class Softmax(Loss):
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_output = False

    # Check if the NN has only a single output (i.e. only one class)
    def _output(self) -> float:
        if self.target.shape[1] == 0:
            self.single_output = True

        # If there is a single output, normalize it
        if self.single_output:
            self.prediction = normalize(self.prediction)
            self.target = normalize(self.target)

        softmax_preds = softmax(self.prediction, axis=1)
        # Clip softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps) 

        softmax_loss = (-1.0 * self.target * np.log(self.softmax_preds) - (1.0 - self.target) * np.log(1 - self.softmax_preds))

        return np.sum(softmax_loss) / self.prediction.shape[0]


    def _input_grad(self) -> np.ndarray:
        if self.single_output:
            return unnormalize(self.softmax_preds - self.target)
        else:
            return (self.softmax_preds - self.target) / self.prediction.shape[0]