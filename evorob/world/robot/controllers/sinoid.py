import numpy as np

from evorob.world.robot.controllers.base import Controller

class OscillatoryController(Controller):
    """Simple oscillatory controller using sine waves for each actuator.

    This controller generates periodic motion patterns without using observations.
    Each joint oscillates with its own amplitude, frequency, and phase.
    """

    def __init__(
        self, input_size: int = 0, output_size: int = None, hidden_size: int = 0
    ):
        """Initialize the oscillatory controller.

        Args:
            output_size: Number of actuators to control
            input_size: Not used, kept for API compatibility
        """
        assert output_size is not None, (
            "output_size must be specified for OscillatoryController"
        )
        self.output_size = output_size
        self.time_step = 0.0
        self.n_params = self.get_num_params()

        self.amplitudes = np.random.uniform(0.1, 1.0, output_size)
        self.frequencies = np.random.uniform(0.5, 2.0, output_size)
        self.phases = np.random.uniform(0, 2 * np.pi, output_size)

    def get_action(self, state):
        """Generate oscillatory actions based on time.

        Args:
            state: Observation (not used by this controller)

        Returns:
            actions: Array of actuator commands, shape (output_size,) or (batch_size, output_size)
        """
        actions = self.amplitudes * np.sin(
            2 * np.pi * self.frequencies * self.time_step + self.phases
        )
        actions = np.clip(actions, -1.0, 1.0)
        self.time_step += 0.01

        if np.ndim(state) == 2:
            batch_size = state.shape[0]
            actions = np.tile(actions, (batch_size, 1))

        return actions

    def set_weights(self, weights):
        """Set controller parameters from flat array.

        Args:
            weights: Flat array of size (3 * output_size,)
                    [amplitudes, frequencies, phases]
        """
        n = self.output_size
        self.amplitudes = weights[0:n]
        self.frequencies = 5 * weights[n:2 * n]
        self.phases = np.pi * weights[2 * n:3 * n]
        self.time_step = 0.0

    def geno2pheno(self, genotype):
        """Alias for set_weights."""
        self.set_weights(genotype)
        self.reset_controller()

    def get_num_params(self):
        """Return total number of parameters.

        Returns:
            int: 3 * output_size (amplitude, frequency, phase for each actuator)
        """
        return 3 * self.output_size

    def reset_controller(self):
        """Reset the controller state (time)."""
        self.time_step = 0.0
