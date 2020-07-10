import libspn as spn
import tensorflow as tf

indicator_leaves = spn.IndicatorLeaf(
    num_vars=2, num_vals=2, name="indicator_x")

# Generate random structure with 1 decomposition per product layer
# 2 subsets of variables per product (so 2 children) and 2 sums/mixtures per scope
dense_spn_generator = spn.DenseSPNGenerator(num_decomps=1, num_subsets=2, num_mixtures=2)
root = dense_spn_generator.generate(indicator_leaves)

# Connect a latent indicator
indicator_y = root.generate_latent_indicators(name="indicator_y") # Can be added manually

# Generate weights
spn.generate_weights(root, initializer=tf.initializers.random_uniform()) # Can be added manually

# Inspect
print(root.get_num_nodes())
print(root.get_scope())
print(root.is_valid())
