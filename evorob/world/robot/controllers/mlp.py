import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        """Initialize a simple feedforward neural network.

        Network structure: input -> hidden -> output
        Activation: tanh on both layers

        Args:
            input_size: Dimension of input (observation size)
            output_size: Dimension of output (action size)
            hidden_size: Number of hidden neurons
        """
        # Here we randomly initialize our neural network layers,
        # as well as our input and output size.
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        self.input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.hidden_to_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))

        self.n_params_i2h = input_size * hidden_size
        self.n_params_h2o = hidden_size * output_size

        self.n_params = self.get_num_params()


    def get_action(self, state):
        """Forward pass through the network.

        Args:
            state: Observation array, shape (input_size,) or (batch_size, input_size)

        Returns:
            action: Output array, shape (output_size,) or (batch_size, output_size)
        """
        hidden = np.tanh(state @ self.input_to_hidden.T)
        output = np.tanh(hidden @ self.hidden_to_output.T)
        return np.clip(output, -1, 1)

    def set_weights(self, encoding):
        """Set network weights from a flat parameter vector.

        Args:
            encoding: Flat array of size (n_params,) containing all weights
        """
        self.input_to_hidden = encoding[:self.n_params_i2h].reshape(self.n_hidden, self.n_input)
        self.hidden_to_output = encoding[self.n_params_i2h:].reshape(self.n_output, self.n_hidden)

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        return self.n_params_i2h + self.n_params_h2o

    def reset_controller(self, batch_size=1) -> None:
        pass
