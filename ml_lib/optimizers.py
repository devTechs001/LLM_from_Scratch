import numpy as np

class GradientDescent:
    def __init__(self,learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Updates a parameter array in the oppsosote direction of its gradient"""

        return param - self.learning_rate * grad