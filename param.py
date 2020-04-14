class Parameter:
    def __init__(self, value):
        """
        A wrapper for parameters. For weights, value has the shape [in_dim, out_dim], whereas
        for biases the shape is [out_dim, ].
        :param value:       (numpy.ndarray) initial values of parameters
        """
        self.value = value
        self.shape = value.shape
        self.grad = None                # store gradient during backpropagation