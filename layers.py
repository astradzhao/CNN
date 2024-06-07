import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self,
        param_name: str,
        X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out,
            activation=activation,
            weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})  # DO NOT CHANGE THE KEYS
        self.cache: OrderedDict = OrderedDict()  # cache for backprop
        self.gradients: OrderedDict = OrderedDict(
            {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        )  # parameter gradients initialized to zero
        # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###

        # perform an affine transformation and activation
        Z = np.dot(X, self.parameters["W"]) + self.parameters["b"]
        out = self.activation(Z)

        # store information necessary for backprop in `self.cache`
        self.cache["Z"] = Z
        self.cache["X"] = X

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        # unpack the cache
        X = self.cache["X"]
        Z = self.cache["Z"]

        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        dLdZ = self.activation.backward(Z, dLdY)

        dLdW = np.dot(X.T, dLdZ)
        dLdB = np.sum(dLdZ, axis=0)
        dX = np.dot(dLdZ, self.parameters["W"].T)

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        self.gradients["W"] = dLdW
        self.gradients["b"] = dLdB

        ### END YOUR CODE ###
        return dX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})  # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []})  # cache for backprop
        self.gradients = OrderedDict(
            {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        )  # parameter gradients initialized to zero
        # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        ### BEGIN YOUR CODE ###

        # implement a convolutional forward pass
        out_rows = 1 + (in_rows + 2 * self.pad[0] - kernel_height) // self.stride
        out_cols = 1 + (in_cols + 2 * self.pad[1] - kernel_width) // self.stride

        X_padded = np.pad(
            X,
            ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)),
            mode="constant",
        )

        shape = (
            n_examples,
            out_rows,
            out_cols,
            kernel_height,
            kernel_width,
            in_channels,
        )
        strides = (
            X_padded.strides[0],
            X_padded.strides[1] * self.stride,
            X_padded.strides[2] * self.stride,
            X_padded.strides[1],
            X_padded.strides[2],
            X_padded.strides[3],
        )

        X_strided = np.lib.stride_tricks.as_strided(
            X_padded, shape=shape, strides=strides
        )

        X_strided = X_strided.reshape(n_examples, out_rows, out_cols, -1)

        Z = np.einsum("ijkl,lm->ijkm", X_strided, W.reshape(-1, out_channels)) + b

        out = self.activation(Z)

        self.cache["Z"] = Z
        self.cache["X"] = X

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        # perform a backward pass
        X = self.cache["X"]
        Z = self.cache["Z"]
        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, _, out_channels = W.shape
        _, in_rows, in_cols, _ = X.shape

        out_rows = 1 + (in_rows + 2 * self.pad[0] - kernel_height) // self.stride
        out_cols = 1 + (in_cols + 2 * self.pad[1] - kernel_width) // self.stride

        dLdZ = self.activation.backward(Z, dLdY)
        dLdZ = dLdZ.transpose(0, 3, 1, 2)

        X_padded = np.pad(
            X,
            ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)),
            mode="constant",
        )

        # transpose elements to put out_channels first
        X_padded = X_padded.transpose(0, 3, 1, 2)
        X = X.transpose(0, 3, 1, 2)
        W = W.transpose(3, 2, 0, 1)

        dX = np.zeros_like(X)
        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(W)
        dB = np.sum(dLdZ, axis=(0, 2, 3))

        for f in range(out_channels):
            for i in range(out_rows):
                for j in range(out_cols):
                    a1 = i * self.stride
                    a2 = i * self.stride + kernel_height
                    b1 = j * self.stride
                    b2 = j * self.stride + kernel_width
                    dW[f] += np.einsum(
                        "ijkl,i->jkl",
                        X_padded[:, :, a1:a2, b1:b2],
                        dLdZ[:, f, i, j],
                    )
                    dX_padded[:, :, a1:a2, b1:b2] += np.einsum(
                        "jkl,i->ijkl", W[f], dLdZ[:, f, i, j]
                    )

        if dX.shape != dX_padded.shape:
            dX = dX_padded[:, :, self.pad[0] : -self.pad[0], self.pad[1] : -self.pad[1]]

        dX = dX.transpose(0, 2, 3, 1)
        dW = dW.transpose(2, 3, 1, 0)

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        return dX


class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        batch_size, in_rows, in_cols, channels = X.shape
        kernel_height, kernel_width = self.kernel_shape
        stride = self.stride

        out_rows = 1 + (in_rows + 2 * self.pad[0] - kernel_height) // self.stride
        out_cols = 1 + (in_cols + 2 * self.pad[1] - kernel_width) // self.stride

        X_padded = np.pad(
            X,
            ((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)),
            mode="constant",
        )

        shape = (batch_size, out_rows, out_cols, kernel_height, kernel_width, channels)
        strides = (
            X_padded.strides[0],
            X_padded.strides[1] * stride,
            X_padded.strides[2] * stride,
            X_padded.strides[1],
            X_padded.strides[2],
            X_padded.strides[3],
        )
        X_strided = np.lib.stride_tricks.as_strided(
            X_padded, shape=shape, strides=strides
        )

        if self.mode == "max":
            X_pool = np.max(X_strided, axis=(3, 4))
        elif self.mode == "average":
            X_pool = np.mean(X_strided, axis=(3, 4))
        else:
            raise ValueError("Unsupported pooling mode. Use 'max' or 'average'.")

        self.cache["X_padded"] = X_padded
        self.cache["X_strided"] = X_strided
        self.cache["X_pool_shape"] = (batch_size, out_rows, out_cols, channels)

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        X_padded = self.cache["X_padded"]
        X_strided = self.cache["X_strided"]
        batch_size, out_rows, out_cols, channels = dLdY.shape
        kernel_height, kernel_width = self.kernel_shape
        stride = self.stride

        dX_padded = np.zeros_like(X_padded)

        if self.mode == "max":
            dLdY_expanded = dLdY[:, :, :, np.newaxis, np.newaxis, :]
            max_mask = X_strided == np.max(X_strided, axis=(3, 4), keepdims=True)
            dX_strided = max_mask * dLdY_expanded
        elif self.mode == "average":
            equal_share = dLdY[:, :, :, np.newaxis, np.newaxis, :] / (
                kernel_height * kernel_width
            )
            dX_strided = np.ones_like(X_strided) * equal_share
        else:
            raise ValueError("Unsupported pooling mode")

        for n in range(batch_size):
            for h in range(out_rows):
                for w in range(out_cols):
                    for c in range(channels):
                        h_start = h * stride
                        h_end = h_start + kernel_height
                        w_start = w * stride
                        w_end = w_start + kernel_width
                        dX_padded[n, h_start:h_end, w_start:w_end, c] += dX_strided[
                            n, h, w, :, :, c
                        ]

        if self.pad[0] > 0 or self.pad[1] > 0:
            dX = dX_padded[:, self.pad[0] : -self.pad[0], self.pad[1] : -self.pad[1], :]
        else:
            dX = dX_padded

        return dX


class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
