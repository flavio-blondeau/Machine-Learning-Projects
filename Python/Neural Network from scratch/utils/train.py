import numpy as np
from typing import Tuple
from copy import deepcopy

from .network import NeuralNetwork
from .optimizers import Optimizer
from .utility_functions import permute_data


class Trainer(object):
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net 
        self.optim = optim
        setattr(self.optim, 'net', self.net) # Set net as an attribute of the optimizer
        self.best_loss = 1e9


    # Generate batches
    def generate_batches(self, X: np.ndarray, y: np.ndarray, size: int = 32) -> Tuple[np.ndarray]:

            assert X.shape[0] == y.shape[0], "Features and target must have the same number of rows, instead features has {} and target has {}".format(X.shape[0], y.shape[0])

            N = X.shape[0]

            for j in range(0, N, size):
                X_batch, y_batch = X[j:j+size], y[j:j+size]

                yield X_batch, y_batch
    

    # Fit the NN on training data for a number of epochs and evaluate it every 'eval_every' epochs
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
            epochs: int = 100, eval_every: int = 10, batch_size: int = 32, seed: int = 42, single_output: bool = False, restart: bool = True, early_stopping: bool = True, conv_testing: bool = False) -> None:
        
        setattr(self.optim, 'max_epochs', epochs)
        self.optim._setup_decay()

        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for t in range(epochs):
        # Early stopping
            if (t+1) % eval_every == 0:
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train,y_train)

            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for j, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

                if conv_testing:
                    if j % 10 == 0:
                        test_preds = self.net.forward(X_batch, inference = True)
                        batch_loss = self.net.loss.forward(test_preds, y_batch)
                        print("batch: ", j, "loss: ", batch_loss)

                    if j % 100 == 0 and j > 0:
                        print("Validation accuracy after", j, "batches is",
                            f"{np.equal(np.argmax(self.net.forward(X_test, inference = True), axis = 1), np.argmax(y_test, axis = 1)).sum() * 100.0 / X_test.shape[0]:.2f}%")

            if (t+1) % eval_every == 0:
                test_preds = self.net.forward(X_test, inference = True)
                loss = self.net.loss.forward(test_preds, y_test)
                
                if early_stopping:
                    if loss < self.best_loss:
                        print(f"Validation loss after {t+1} epochs is {loss:.3f}.")
                        self.best_loss = loss

                    else:
                        print()
                        print(f"Loss increased after epoch {t+1}, the final loss was {self.best_loss:.3f}, using the model from epoch {t+1-eval_every}")
                        self.net = last_model
                
                        setattr(self.optim, 'net', self.net) # Ensure self.optim still update self.net
                        break

                else:
                    print(f"Validation loss after {t+1} epochs is {loss:.3f}")

            if self.optim.final_lr:
                self.optim._decay_lr()