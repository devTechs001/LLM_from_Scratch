import numpy as np
from ml_lib.losses import mse_loss, mse_gradient
from ml_lib.optimizers import GradientDescent

#Dummy target and initial prediction
y_true = np.array([10.0,20.0,30.0])
y_pred = np.array([2.0,2.0,2.0])

optimizer = GradientDescent(learning_rate=0.1)

print("Initial MSE Loss:",mse_loss(y_true,y_pred))

#Run 10 optimization steps directly on y_pred

for _ in range(10):
    grad = mse_gradient(y_true,y_pred)
    y_pred = optimizer.step(y_pred, grad)

print("Updated predictions:", np.round(y_pred,2))
print("FInal MSE Loss:", round(mse_loss(y_true, y_pred)))
