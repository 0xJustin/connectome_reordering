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

