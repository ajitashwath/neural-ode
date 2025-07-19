import jax.numpy as jnp
import jax.random as jrandom

def get_data(dataset_name, data_size, key):
    if dataset_name == "spiral":
        return make_spiral(data_size, key)
    elif dataset_name == "circle":
        return make_concentric_circles(data_size, key)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def make_spiral(data_size, key):
    n = data_size;
    key, r_key, theta_key, noise_key = jrandom.split(key, 4)

    theta0 = jnp.sqrt(jrandom.uniform(r_key, (n, 1))) * 3 * jnp.pi
    r0 = theta0 / (3 * jnp.pi) + 0.05
    x0 = jnp.append(r0 * jnp.cos(theta0), r0 * jnp.sin(theta0), axis = 1)
    y0 = jnp.zeros((n, 1))

    theta1 = jnp.sort(jrandom.uniform(theta_key, (n, 1))) * 3 * jnp.pi
    r1 = theta1 / (3 * jnp.pi) + 0.05
    x1 = jnp.append(-r1 * jnp.cos(theta1), -r1 * jnp.sin(theta1), axis = 1)
    y1 = jnp.ones((n, 1))

    x = jnp.concatenate([x0, x1])
    y = jnp.concatenate([y0, y1]).ravel()

    x += jrandom.normal(noise_key, x.shape) * 0.1

    ts = jnp.array([0.0, 1.0])
    ys = (x, y)
    return ts, ys, key

def make_concentric_circles(data_size, key):
    n = data_size
    key, angle_key1, angle_key2, noise_key1, noise_key2 = jrandom.split(key, 5)

    radius0 = 1.0
    angles0 = jrandom.uniform(angle_key1, (n,)) * 2 * jnp.pi
    x0 = radius0 * jnp.cos(angles0)
    y0_coord = radius0 * jnp.sin(angles0)
    class0 = jnp.stack([x0, y0_coord], axis=1)
    class0 += jrandom.normal(noise_key1, class0.shape) * 0.1
    labels0 = jnp.zeros(n)

    radius1 = 2.0
    angles1 = jrandom.uniform(angle_key2, (n,)) * 2 * jnp.pi
    x1 = radius1 * jnp.cos(angles1)
    y1_coord = radius1 * jnp.sin(angles1)
    class1 = jnp.stack([x1, y1_coord], axis=1)
    class1 += jrandom.normal(noise_key2, class1.shape) * 0.1
    labels1 = jnp.ones(n)

    x = jnp.concatenate([class0, class1])
    y = jnp.concatenate([labels0, labels1])

    ts = jnp.array([0.0, 1.0])
    ys = (x, y)
    return ts, ys, key
