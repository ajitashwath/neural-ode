import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt

class ODEFunc(eqx.Module):
    mlq: eqx.nn.NLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size = 2,
            out_size = 2,
            width_size = 64,
            depth = 2,
            activation = jnp.tanh,
            key = key
        )

    def __call__(self, t, y, args):
        return self.mlp(y)
    
class NeuralODE(eqx.Module):
    func: ODEFunc
    solver: Dopri5

    def __init__(self, key):
        self.func = ODEFunc(key)
        self.solver = Dopri5()

    def __call__(self, y0, ts):
        solution = diffeqsolve(
            terms = ODETerm(self.func),
            solver = self.solver,
            t0 = ts[0],
            t1 = ts[-1],
            dt0 = ts[1] - ts[0],
            y0 = y0,
            saveat = SaveAt(ts = ts)
        )
        return solution.ys