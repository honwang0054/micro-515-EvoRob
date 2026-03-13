import numpy as np

from evorob.algorithms.base_ea import EA


class EvoAlgAPI(EA):
    """Evolutionary algorithm API wrapper.

    This class provides an interface to wrap any EA framework that uses
    the ask-tell pattern (CMA-ES, pyribs, evosax, etc.).

    Example frameworks to use:
    - CMA-ES: https://github.com/CMA-ES/pycma
    - pyribs: https://github.com/icaros-usc/pyribs/
    - evosax: https://github.com/RobertTLange/evosax/
    - EvoJAX: https://github.com/google/evojax
    """

    def __init__(self, n_params: int, population_size: int = 100, num_generations: int = 100,
                 output_dir: str = "./results/EA", **kwargs):
        """Initialize the evolutionary algorithm.

        Args:
            n_params: Dimensionality of the search space
            population_size: Number of solutions per generation
            num_generations: Number of generations
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments for the EA framework
        """
        self.n_params = n_params
        self.n_gen = num_generations
        self.population_size = population_size
        self.framework = kwargs.get("framework", "cma")
        
        if self.framework == "cma":
            import cma
            self.es = cma.CMAEvolutionStrategy(self.n_params * [0.0], 0.1, {'popsize': self.population_size})
        elif self.framework == "pyribs":
            from ribs.archives import GridArchive
            from ribs.emitters import GaussianEmitter
            from ribs.optimizers import Optimizer
            self.archive = GridArchive(solution_dim=self.n_params, dims=[1], ranges=[(-1.0, 1.0)])
            self.emitters = [GaussianEmitter(self.archive, x0=np.zeros(self.n_params), sigma=0.5, batch_size=self.population_size)]
            self.optimizer = Optimizer(self.archive, self.emitters)
        elif self.framework == "evosax":
            import jax
            from evosax.algorithms import CMA_ES
            self.strategy = CMA_ES(popsize=self.population_size, num_dims=self.n_params)
            self.es_params = self.strategy.default_params
            self.es_state = self.strategy.initialize(jax.random.PRNGKey(0), self.es_params)
        elif self.framework == "evojax":
            from evojax.algorithms import PGPE
            self.solver = PGPE(pop_size=self.population_size, param_size=self.n_params, optimizer='adam', center_learning_rate=0.01, stdev_learning_rate=0.1)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
            
        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

    def ask(self) -> np.ndarray:
        """Sample population from the algorithm.

        Returns:
            population: Array of shape (population_size, n_params)
                       Each row is a candidate solution
        """
        if self.framework == "cma":
            self._current_pop = self.es.ask()
            return np.array(self._current_pop)
        elif self.framework == "pyribs":
            self._current_pop = self.optimizer.ask()
            return np.array(self._current_pop)
        elif self.framework == "evosax":
            import jax
            rng = jax.random.PRNGKey(self.current_gen)
            x, self.es_state = self.strategy.ask(rng, self.es_state, self.es_params)
            self._current_pop = np.array(x)
            return self._current_pop
        elif self.framework == "evojax":
            self._current_pop = np.array(self.solver.ask())
            return self._current_pop

    def tell(self, population: np.ndarray, fitnesses: np.ndarray, save_checkpoint: bool = False) -> None:
        """Update the algorithm with evaluated population.

        Args:
            population: Array of shape (population_size, n_params)
            fitnesses: Array of shape (population_size,) with fitness values
                      Higher is better (maximization)
            save_checkpoint: Whether to save checkpoint after update
        """
        if self.framework == "cma":
            self.es.tell(self._current_pop, -fitnesses)
        elif self.framework == "pyribs":
            measures = np.zeros((self.population_size, 1))
            self.optimizer.tell(fitnesses, measures)
        elif self.framework == "evosax":
            import jax.numpy as jnp
            self.es_state = self.strategy.tell(jnp.array(population), -jnp.array(fitnesses), self.es_state, self.es_params)
        elif self.framework == "evojax":
            import jax.numpy as jnp
            self.solver.tell(jnp.array(fitnesses))
            
        # After updating the EA, do bookkeeping for checkpointing:
        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population
        
        # Track best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()
        
        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1
