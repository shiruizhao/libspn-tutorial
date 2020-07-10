
#Tutorial 1b: Building the SPN Graph Using Multi-Op Nodes

#We can also build SPNs using bigger building blocks: e.g. ParallelSums which compute multiple sums with identical child inputs or PermuteProducts which compute all permutations of children with disjoint scopes.

import libspn as spn
import tensorflow as tf

#Build the SPN

indicator_leaves = spn.IndicatorLeaf(
    num_vars=2, num_vals=2, name="indicator_x")

# Connect first two sums to indicators of first variable
sums_1 = spn.ParallelSums((indicator_leaves, [0,1]), num_sums=2, name="sums_1")
# Connect another two sums to indicators of second variable
sums_2 = spn.ParallelSums((indicator_leaves, [2,3]), num_sums=2, name="sums_2")

# Connect 2 * 2 == 4 product nodes
prods_1 = spn.PermuteProducts(sums_1, sums_2, name="prod_1")

# Connect a root sum
root = spn.Sum(prods_1, name="root")

# Connect a latent indicator
indicator_y = root.generate_latent_indicators(name="indicator_y") # Can be added manually

# Generate weights
spn.generate_weights(root, initializer=tf.initializers.random_uniform()) # Can be added manually


## Inspect

# Inspect
print(root.get_num_nodes())
print(root.get_scope())
print(root.is_valid())

#Visualize the SPN Graph

#The visualization below uses graphviz. Depending on your setup (e.g. jupyter lab vs. jupyter notebook) this might fail to show. At least Chrome + jupyter notebook seems to work.

# Visualize SPN graph
spn.display_spn_graph(root)


