import libspn as spn
import tensorflow as tf

spn.config_logger(spn.DEBUG1)

indicator_x = spn.IndicatorLeaf(
    num_vars=2, num_vals=2, name="indicator_x")

# Build structure and attach weight
sum_11 = spn.Sum((indicator_x, [0,1]), name="sum_11")
sum_11.generate_weights(initializer=tf.initializers.constant([0.4, 0.6]))
sum_12 = spn.Sum((indicator_x, [2,3]), name="sum_12")
sum_12.generate_weights(initializer=tf.initializers.constant([0.1, 0.9]))
sum_21 = spn.Sum((indicator_x, [2,3]), name="sum_21")
sum_21.generate_weights(initializer=tf.initializers.constant([0.7, 0.3]))
sum_22 = spn.Sum((indicator_x, [2,3]), name="sum_22")
sum_22.generate_weights(initializer=tf.initializers.constant([0.8, 0.2]))
prod_1 = spn.Product(sum_11, sum_21, name="prod_1")
prod_2 = spn.Product(sum_11, sum_22, name="prod_2")
prod_3 = spn.Product(sum_12, sum_22, name="prod_3")
root = spn.Sum(prod_1, prod_2, prod_3, name="root")
root.generate_weights(initializer=tf.initializers.constant([0.5, 0.2, 0.3]))

# Connect a latent indicator
indicator_y = root.generate_latent_indicators(name="indicator_y") # Can be added manually

# Inspect
print(root.get_num_nodes())
print(root.get_scope())
print(root.is_valid())

init_weights = spn.initialize_weights(root)
marginal_val = root.get_value(inference_type=spn.InferenceType.MPE)
sum_11_val = sum_11.get_value(inference_type=spn.InferenceType.MPE)
sum_12_val = sum_12.get_value(inference_type=spn.InferenceType.MPE)
sum_21_val = sum_21.get_value(inference_type=spn.InferenceType.MPE)
sum_22_val = sum_22.get_value(inference_type=spn.InferenceType.MPE)
prod_1_val = prod_1.get_value(inference_type=spn.InferenceType.MPE)
prod_2_val = prod_2.get_value(inference_type=spn.InferenceType.MPE)
prod_3_val = prod_3.get_value(inference_type=spn.InferenceType.MPE)
indicator_y_val = indicator_y.get_value(inference_type=spn.InferenceType.MPE)
mpe_val = root.get_value(inference_type=spn.InferenceType.MPE)

indicator_x_data = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 0],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1]
]

indicator_y_data = [[0], [2], [0], [0], [-1], [-1], [-1], [1], [0]]

with tf.Session() as sess:
    init_weights.run()
    marginal_val_arr = sess.run(marginal_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data})
    mpe_val_arr = sess.run(mpe_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data})
    #print(sess.run(sum_11_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    #print(sess.run(sum_12_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    #print(sess.run(sum_21_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    #print(sess.run(sum_22_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    print(sess.run(prod_1_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    print(sess.run(prod_2_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    print(sess.run(prod_3_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    print(sess.run(indicator_y_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))
    print(sess.run(mpe_val, feed_dict={indicator_x: indicator_x_data, indicator_y: indicator_y_data}))

#print(marginal_val_arr)
#print(mpe_val_arr)
