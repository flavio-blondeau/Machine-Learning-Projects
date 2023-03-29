import numpy as np


class Optimizer(object):

    def __init__(self, learning_rate: float = 0.01, final_lr: float = 0, decay_type: str = None) -> None:
        self.learning_rate = learning_rate
        self.final_lr = final_lr
        self.decay_type = decay_type # Possible types: None, linear and exponential
        self.first = True


    def _setup_decay(self) -> None:

        if not self.decay_type:
            return
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.learning_rate - self.final_lr) / (self.max_epochs - 1)
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_lr / self.learning_rate, 1.0 / (self.max_epochs - 1))


    def _decay_lr(self) -> None:

        if not self.decay_type:
            return
        
        if self.decay_type == 'linear':
            self.learning_rate -= self.decay_per_epoch

        if self.decay_type == 'exponential':
            self.learning_rate *= self.decay_per_epoch
    

    # This method must be impelmented for each Optimizer
    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError


    def step(self, epoch: int = 0) -> None:

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param = param, grad = param_grad)



class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, final_lr: float = 0, decay_type: str = None) -> None:
        super().__init__(learning_rate, final_lr, decay_type)


    def _update_rule(self, **kwargs) -> None:
        
        update = self.learning_rate * kwargs['grad']
        kwargs['param'] -= update



class SGDMomentum(Optimizer):

    def __init__(self, learning_rate: float = 0.01, final_lr: float = 0, decay_type: str = None, momentum: float = 0.9) -> None:
        super().__init__(learning_rate, final_lr, decay_type)
        self.momentum = momentum

    
    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(), self.net.param_grads(), self.velocities):
            self._update_rule(param = param, grad = param_grad, velocity = velocity)


    def _update_rule(self, **kwargs) -> None:
        
        kwargs['velocity'] *= self.momentum
        kwargs['velocity'] += self.learning_rate * kwargs['grad']

        kwargs['param'] -= kwargs['velocity']