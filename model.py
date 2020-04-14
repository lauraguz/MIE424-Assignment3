class NeuralNetwork:
    def __init__(self):
        """
        A module that combines all the components of a neural network.
        """
        self.layers = []        # FCLayer and activation layers will be appended
        self._parameters = []
        self.loss = None        # the loss function
        self.diff_loss = None   # and its derivative
        self.optimizer = None

    def set_optimizer(self, opt):
        """
        Method for setting optimizer to use.
        You should instantiate an Optimizer object and set_optimizer() it before taking any training steps.
        :param opt:         (optim.Optimizer)
        """
        self.optimizer = opt

    def add(self, layer):
        """
        Append layer and layer.param (if exists) to self.layer and self._parameters, respectively.
        :param layer:       (Layer.Layer) a Layer object to be appended to self.layers
        """
        self.layers.append(layer)

        # FCLayer object has Parameter object stored as layer.param
        try:
            self._parameters.extend(layer.param)
        # no parameters in Activation object
        except:
            pass

    def parameters(self):
        """
        Getter for parameters in the network
        :return:            (list) self._parameters (a list of parameters of each layer)
        """
        return self._parameters

    def set_loss(self, loss):
        """
        Method for setting loss function and its derivative.
        :param loss:        (Loss) a Loss object that has 'loss' and 'diff_loss'
        """
        self.loss = loss.loss
        self.diff_loss = loss.diff_loss

    def predict(self, input_data, mode=True):
        """
        :param input_data:  (numpy.ndarray) an array of input samples with the shape [n x m_0].
        :param mode:        (bool) set to True during forward pass in training; False when testing.
        :return:            (numpy.ndarray) the output from the output layer.
        """
        out = input_data

        # Feed forward
        for l in self.layers:
            out = l.forward(out, mode)
        return out

    def backward(self, pred, y_true):
        """
        Backpropagate errors to inner layers.
        :param pred:        (numpy.ndarray) the output of self.predict(input)
        :param y_true:      (numpy.ndarray) true targets of data
        """
        # Take derivative of the loss function
        delta = self.diff_loss(pred, y_true)

        # Backpropagate the errors
        for l in reversed(self.layers):
            delta = l.backward(delta)
