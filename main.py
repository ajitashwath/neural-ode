import jax
import jax.random as jrandom
import matplotlib.pyplot as plt
import equinox as eqx
import jax.numpy as jnp
from diffrax import Dopri5

from model import NeuralODE, ODEFunc
from data_loader import get_data
from training import train

def main():
    learning_rate = 3e-3
    num_steps = 2000
    dataset_name = "spiral"
    data_size = 1000
    batch_size = 256
    seed = 42

    key = jrandom.PRNGKey(seed)
    data_key, model_key, train_key = jrandom.split(key, 3)
    ts, ys, data_key = get_data(dataset_name, data_size, data_key)

    ode_func_mlp = eqx.nn.MLP(
        in_size = 2,
        out_size = 2,
        width_size = 64,
        depth = 2,
        activation = jnp.tanh,
        key = model_key
    )
    ode_func = ODEFunc(mlp = ode_func_mlp)
    model = NeuralODE(func = ode_func, solver = Dopri5())
    
    print("Starting training...")
    model, loss_history = train(model, ts, ys, num_steps, batch_size, learning_rate, train_key)

    plt.figure(figsize = (10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()