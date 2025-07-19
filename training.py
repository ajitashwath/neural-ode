import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from model import NeuralODE

@eqx.filter_value_and_grad
def grad_loss(model, x, y, ts):
    y_pred = jax.vmap(model, in_axes = (0, None))(x, ts) # Basic loss
    logits = y_pred[:, 1, 0]
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()

def train(model: NeuralODE, ts, ys, num_steps, batch_size, learning_rate, key):
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    x_all, y_all = ys
    dataset_size = x_all.shape[0]
    
    loss_history = []
    for step in range(num_steps):
        key, subkey = jrandom.split(key)
        indices = jrandom.choice(subkey, dataset_size, (batch_size,), replace=False)
        x_batch = x_all[indices]
        y_batch = y_all[indices]

        loss, grads = grad_loss(model, x_batch, y_batch, ts)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        loss_history.append(loss.item())
        if step % 100 == 0 or step == num_steps - 1:
            print(f"Step: {step}, Loss: {loss.item():.4f}")
    visualize_dynamics(model, ys, ts)
    
    return model, loss_history

def visualize_dynamics(model, ys, ts):
    x_data, y_data = ys

    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, 20), jnp.linspace(y_min, y_max, 20))
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    dz_dt = jax.vmap(model.func, in_axes=(None, 0, None))(0.0, grid_points, None)

    num_trajectories = 50
    trajectory_ts = jnp.linspace(ts[0], ts[-1], 50)
    trajectories = jax.vmap(model, in_axes=(0, None))(x_data[:num_trajectories], trajectory_ts)
    plt.figure(figsize=(10, 10))

    plt.quiver(grid_points[:, 0], grid_points[:, 1], dz_dt[:, 0], dz_dt[:, 1], color="gray", alpha=0.6)

    for i in range(num_trajectories):
        plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], '-', lw=1, color='blue', alpha=0.5)
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='viridis', edgecolors='k')
    
    plt.title("Learned Dynamics and Trajectories")
    plt.xlabel("z_1")
    plt.ylabel("z_2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.show()