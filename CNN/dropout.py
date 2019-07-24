import numpy as np
class Dropout(object):
    def __init__(self,  p):
        """
        A dropout regularization wrapper.

        During training, independently zeroes each element of the layer input
        with probability p and scales the activation by 1 / (1 - p) (to reflect
        the fact that on average only (1 - p) * N units are active on any
        training pass). At test time, does not adjust elements of the input at
        all (ie., simply computes the identity function).

        Parameters
        ----------
        wrapped_layer : `layers.LayerBase` instance
            The layer to apply dropout to.
        p : float in [0, 1)
            The dropout propbability during training
        """
        self.p = p
    
    def forward(self, X,trainable=True):
        self.trainable=trainable
        scaler, mask = 1.0, np.ones(X.shape).astype(bool)
        if trainable:
            scaler = 1.0 / (1.0 - self.p)
            mask = np.random.rand(*X.shape) >= self.p
            X = mask * X
        self.mask = mask
        return scaler * X

    def backward(self, dLdy):
        assert self.trainable, "Layer is frozen"
        dLdy=self.mask*dLdy
        dLdy *= 1.0 / (1.0 - self.p)
        return dLdy