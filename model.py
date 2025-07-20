import equinox as eqx
import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt

class ODEFunc(eqx.Module):
    """
    Represents the function f(z, t; theta) in the Neural ODE.
    
    This class now correctly defines `mlp` as a field. Equinox will
    automatically create an __init__ method that accepts `mlp` as an argument.
    The old __init__ method has been removed.
    """
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.mlp(y)

class NeuralODE(eqx.Module):
    """
    Defines the Neural Ordinary Differential Equation model.

    This class correctly defines `func` and `solver` as fields. Equinox
    will automatically create an __init__ that accepts these as arguments.
    """
    func: ODEFunc
    solver: Dopri5

    def __call__(self, y0, ts):
        """
        Solves the ODE from y0 at times ts.
        """
        solution = diffeqsolve(
            terms=ODETerm(self.func),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            saveat=SaveAt(ts=ts)
        )
        return solution.ys