import jax
import jax.numpy as jnp
import optax
from jax import random, jit


def normalize_positions(positions):
    # Normalize positions to have zero mean and unit variance
    mean = jnp.mean(positions)
    std = jnp.std(positions) + 1e-8  # Avoid division by zero
    positions = (positions - mean) / std
    return positions


def calculate_node_forward(source_orders, target_orders, edge_weights):
    forward_edges = source_orders < target_orders
    negative_edges = source_orders > target_orders
    zero_edges = source_orders == target_orders

    # Calculate total forward edge weight
    forward_edge_weight = jnp.sum(edge_weights * forward_edges)
    negative_edges_weight = jnp.sum(edge_weights * negative_edges)
    zero_edges_weight = jnp.sum(edge_weights * zero_edges)

    # Calculate total edge weight (for normalization)
    total_edge_weight = jnp.sum(edge_weights)
    total_edge_weight_negative = jnp.sum(negative_edges_weight)
    total_edge_weight_zero = jnp.sum(zero_edges_weight)

    # Calculate percentage of forward edge weight
    percentage_forward = 100 * (forward_edge_weight / total_edge_weight)
    percentage_negative = 100 * (total_edge_weight_negative / total_edge_weight)
    percentage_zero = 100 * (total_edge_weight_zero / total_edge_weight)

    #print(
    #    f"Percentage of forward edge weight: {percentage_forward:.2f}%, negative edge weight: {percentage_negative:.2f}%, zero edge weight: {percentage_zero:.2f}%"
    #)
    return percentage_forward


def calculate_metric(
    positions, num_nodes, source_indices, target_indices, edge_weights
):
    # Get final positions
    final_positions = positions  # jnp.dot(positions, w)

    # Sort node indices based on positions
    sorted_indices = jnp.argsort(final_positions)
    node_order = jnp.zeros(num_nodes)
    node_order = node_order.at[sorted_indices].set(jnp.arange(num_nodes))

    source_order = node_order[source_indices]
    target_order = node_order[target_indices]
    return calculate_node_forward(source_order, target_order, edge_weights)


@jax.jit
def objective_function(
    relu_weight, positions, beta, source_indices, target_indices, edge_weights,
):
    # Project each neuron embedding onto the learnable direction w
    proj_source = positions[source_indices]
    proj_target = positions[target_indices]
    delta = proj_source - proj_target
    
    # delta = delta / jnp.linalg.norm(delta)

    sigmoid = jax.nn.sigmoid(beta * delta) #* jax.nn.sigmoid(delta * 10000)
    relu = 0#jax.nn.relu(delta)# - (relu_weight))
    # reg = 100 * -jnp.var(positions)
    total_forward_weight = jnp.sum(edge_weights * (sigmoid + relu_weight * relu))
    return total_forward_weight #+ reg


# Function to compute total forward edge weight given an ordering
def compute_total_forward_weight(
    ordering, source_indices, target_indices, edge_weights_normalized
):
    node_ranks = jnp.zeros_like(ordering)
    node_ranks = node_ranks.at[ordering].set(jnp.arange(len(ordering)))
    edge_directions = node_ranks[target_indices] - node_ranks[source_indices]
    forward_edges = edge_directions > 0
    total_forward_weight = jnp.sum(edge_weights_normalized * forward_edges)
    return total_forward_weight



import jax
import jax.numpy as jnp

def monte_carlo_node_ordering(source_indices, target_indices, node_order, edge_weights, num_iterations=200000000, temp=1.0):
    num_nodes = node_order.shape[0]
    num_edges = source_indices.shape[0]

    source_indices = jnp.array(source_indices)
    target_indices = jnp.array(target_indices)
    node_order = node_order.astype(float)

    # Function to compute the forward score
    def calculate_forward_score(node_order):
        source_order = node_order[source_indices]
        target_order = node_order[target_indices]
        forward_edges = source_order < target_order
        return jnp.sum(edge_weights * forward_edges)

    # Initial score
    current_score = calculate_forward_score(node_order)

    # Initial PRNGKey
    key = jax.random.PRNGKey(0)

    def monte_carlo_step(state):
        node_order, current_score, iteration, temp, key = state

        # Split the key for reproducibility
        key, subkey_i, subkey_j, subkey_accept = jax.random.split(key, 4)

        # Sample two random nodes to swap
        i = jax.random.randint(subkey_i, (), 0, num_nodes)
        j = jax.random.randint(subkey_j, (), 0, num_nodes)

        # Ensure that i != j using jax.lax.while_loop
        def cond_fun(val):
            _, j = val
            return j == i

        def body_fun(val):
            key_j, _ = val
            key_j, subkey_new_j = jax.random.split(key_j)
            new_j = jax.random.randint(subkey_new_j, (), 0, num_nodes)
            return key_j, new_j

        key_j, j = jax.lax.while_loop(cond_fun, body_fun, (key, j))

        # Update key with the latest key_j
        key = key_j

        # Swap positions of node i and node j
        new_node_order = node_order.at[i].set(node_order[j])
        new_node_order = new_node_order.at[j].set(node_order[i])

        # Compute new score
        new_score = calculate_forward_score(new_node_order)

        # Acceptance probability (simulated annealing)
        delta = new_score - current_score
        accept_prob = jnp.exp(delta / temp)  # Probabilistically accept worse solutions based on temperature

        random_value = jax.random.uniform(subkey_accept, ())

        # Update state based on acceptance criterion
        accept_swap = (delta > 0) #| (random_value < accept_prob)

        node_order = jax.lax.select(accept_swap, new_node_order, node_order)
        current_score = jax.lax.select(accept_swap, new_score, current_score)

        return node_order, current_score, iteration + 1, temp, key

    # Initial state for the Monte Carlo loop
    state = (node_order, current_score, 0, temp, key)

    # Loop for a given number of iterations
    def cond_fun(state):
        node_order, _, iteration, _, _ = state
        def check_metric_fn(_):
            source_order = node_order[source_indices]
            target_order = node_order[target_indices]
            metric = calculate_node_forward(source_order, target_order, edge_weights)
            jax.debug.print("Iteration {iteration}, metric {metric}, score {current_score}", iteration=iteration, metric=metric, current_score=current_score)
            return None

        # Conditionally run the metric check
        jax.lax.cond(iteration % 100000 == 0, check_metric_fn, lambda _: None, operand=None)

        return iteration < num_iterations

    state = jax.lax.while_loop(cond_fun, monte_carlo_step, state)
    final_node_order, final_score, _, _, _ = state

    return final_node_order, final_score