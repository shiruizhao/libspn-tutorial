#!/usr/bin/env python
# coding: utf-8

# 
# <h1 id="Wicker-Convolutional-SPNs-for-continuous-MNIST-data">Wicker Convolutional SPNs for continuous MNIST data<a class="anchor-link" href="#Wicker-Convolutional-SPNs-for-continuous-MNIST-data">¶</a></h1><p>This notebook shows how to build Wicker Convolutional SPNs (WCSPNs) and use them to classifiy digits with the MNIST dataset.</p>
# <h3 id="Setting-up-the-imports-and-preparing-the-data">Setting up the imports and preparing the data<a class="anchor-link" href="#Setting-up-the-imports-and-preparing-the-data">¶</a></h3><p>We load the data from <code>tf.keras.datasets</code>. Preprocessing consists of flattening and binarization of the data.</p>
# 

# In[1]:



import libspn as spn
import tensorflow as tf
import numpy as np
from libspn.examples.utils.dataiterator import DataIterator

# Load
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data("/home/ray/Downloads/mnist.npz")

def binarize(x):
    return x / 255.

def flatten(x):
    return x.reshape(-1, np.prod(x.shape[1:]))

def preprocess(x, y):
    return binarize(flatten(x)), np.expand_dims(y, axis=1)

# Preprocess
train_x, train_y = preprocess(train_x, train_y)
test_x, test_y = preprocess(test_x, test_y)


# 
# <h3 id="Defining-the-hyperparameters">Defining the hyperparameters<a class="anchor-link" href="#Defining-the-hyperparameters">¶</a></h3><p>Some hyperparameters for the SPN.</p>
# <ul>
# <li><code>num_vars</code> corresponds to the number of input variables (the number of pixels in the case of MNIST).</li>
# <li><code>num_leaf_components</code> is the number of distribution components in the normal leafs</li>
# <li><code>inference_type</code> determines the kind of forward inference where <code>spn.InferenceType.MARGINAL</code> corresponds to sum nodes marginalizing their inputs. <code>spn.InferenceType.MPE</code> would correspond to having max nodes instead.</li>
# <li><code>learning_rate</code> is the learning rate for the Adam optimizer</li>
# <li><code>scale_init</code>, initial scale value for the <code>NormalLeaf</code> node. This parameter greatly determines the stability of the training process</li>
# <li><code>num_classes</code>, <code>batch_size</code> and <code>num_epochs</code> should be obvious:)</li>
# </ul>
# 

# In[2]:



# Number of variables
num_vars = train_x.shape[1]
# Number of different values at leaf (binary here, so 2)
num_leaf_components = 4
# Inference type (can also be spn.InferenceType.MPE) where 
# sum nodes are turned into max nodes
inference_type = spn.InferenceType.MARGINAL
# Adam optimizer parameters
learning_rate = 1e-2
# Scale init
scale_init = 0.1
# Other params
num_classes = 10
batch_size = 32
num_epochs = 50


# 
# <h3 id="Building-the-SPN">Building the SPN<a class="anchor-link" href="#Building-the-SPN">¶</a></h3><p>Our SPN consists of a leaf node with normal distributions followed by spatial products and sums. A <code>ConvProducts</code> node will generate all possible permutations of the child channels (if possible). A <code>ConvProductsDepthwise</code>
#  will use the subset of permutations that corresponds to depthwise 
# convolutions. Products are in fact implemented as convolutions, since 
# multiplications become sums in the log-space. <code>LocalSums</code> consist of sums that are applied 'locally', without weight sharing, so they are in a sense comparable to <code>LocallyConnected</code> layers in <code>Keras</code>.</p>
# <p>Note that after two non-overlapping products (with kernel sizes of <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" data-mathml='&lt;math xmlns="http://www.w3.org/1998/Math/MathML"&gt;&lt;mn&gt;2&lt;/mn&gt;&lt;mo&gt;&amp;#x00D7;&lt;/mo&gt;&lt;mn&gt;2&lt;/mn&gt;&lt;/math&gt;' id="MathJax-Element-1-Frame" role="presentation" style="position: relative;" tabindex="0"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-1" style="width: 2.925em; display: inline-block;"><span style="display: inline-block; position: relative; width: 2.219em; height: 0px; font-size: 132%;"><span style="position: absolute; clip: rect(1.39em, 1002.17em, 2.381em, -1000em); top: -2.219em; left: 0em;"><span class="mrow" id="MathJax-Span-2"><span class="mn" id="MathJax-Span-3" style="font-family: MathJax_Main;">2</span><span class="mo" id="MathJax-Span-4" style="font-family: MathJax_Main; padding-left: 0.222em;">×</span><span class="mn" id="MathJax-Span-5" style="font-family: MathJax_Main; padding-left: 0.222em;">2</span></span><span style="display: inline-block; width: 0px; height: 2.219em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.071em; border-left: 0px solid; width: 0px; height: 1.022em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><mn>2</mn><mo>×</mo><mn>2</mn></math></span></span><script id="MathJax-Element-1" type="math/tex">2\times 2</script> and strides of <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" data-mathml='&lt;math xmlns="http://www.w3.org/1998/Math/MathML"&gt;&lt;mn&gt;2&lt;/mn&gt;&lt;mo&gt;&amp;#x00D7;&lt;/mo&gt;&lt;mn&gt;2&lt;/mn&gt;&lt;/math&gt;' id="MathJax-Element-2-Frame" role="presentation" style="position: relative;" tabindex="0"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-6" style="width: 2.925em; display: inline-block;"><span style="display: inline-block; position: relative; width: 2.219em; height: 0px; font-size: 132%;"><span style="position: absolute; clip: rect(1.39em, 1002.17em, 2.381em, -1000em); top: -2.219em; left: 0em;"><span class="mrow" id="MathJax-Span-7"><span class="mn" id="MathJax-Span-8" style="font-family: MathJax_Main;">2</span><span class="mo" id="MathJax-Span-9" style="font-family: MathJax_Main; padding-left: 0.222em;">×</span><span class="mn" id="MathJax-Span-10" style="font-family: MathJax_Main; padding-left: 0.222em;">2</span></span><span style="display: inline-block; width: 0px; height: 2.219em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.071em; border-left: 0px solid; width: 0px; height: 1.022em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><mn>2</mn><mo>×</mo><mn>2</mn></math></span></span><script id="MathJax-Element-2" type="math/tex">2\times 2</script>), we have a 'wicker' stack where we use <code>'full'</code> padding and exponentially increasing dilation rates.</p>
# <p>Finally, we apply a <code>ConvProductDepthwise</code> layer with <code>'wicker_top'</code>
#  padding to get scopes which include all variables at the final layer. 
# This layer can then be connected to class roots, which are in turn 
# connected to a single root node.</p>
# 

# In[3]:



tf.reset_default_graph()
# Leaf nodes

normal_leafs = spn.NormalLeaf(
    num_components=num_leaf_components, num_vars=num_vars, 
    trainable_scale=False, trainable_loc=True, scale_init=scale_init)

# Twice non-overlapping convolutions
x = spn.ConvProducts(normal_leafs, num_channels=32, padding='valid', kernel_size=2, strides=2, spatial_dim_sizes=[28, 28])
x = spn.LocalSums(x, num_channels=32)
x = spn.ConvProductsDepthwise(x, padding='valid', kernel_size=2, strides=2)
x = spn.LocalSums(x, num_channels=32)

# Make a wicker stack
stack_size = int(np.ceil(np.log2(28 // 4)))
for i in range(stack_size):
    dilation_rate = 2 ** i
    x = spn.ConvProductsDepthwise(
        x, padding='full', kernel_size=2, strides=1, dilation_rate=dilation_rate)
    x = spn.LocalSums(x, num_channels=64)
# Create final layer of products
full_scope_prod = spn.ConvProductsDepthwise(
    x, padding='wicker_top', kernel_size=2, strides=1, dilation_rate=2 ** stack_size)
class_roots = spn.ParallelSums(full_scope_prod, num_sums=num_classes)
root = spn.Sum(class_roots)

# Add a IndicatorLeaf node to the root as a latent class variable
class_indicators = root.generate_latent_indicators()

# Generate the weights for the SPN rooted at `root`
spn.generate_weights(root, log=True, initializer=tf.initializers.random_uniform())

print("SPN depth: {}".format(root.get_depth()))
print("Number of products layers: {}".format(root.get_num_nodes(node_type=spn.ConvProducts)))
print("Number of sums layers: {}".format(root.get_num_nodes(node_type=spn.LocalSums)))


# In[4]:


spn.display_tf_graph()


# 
# <h3 id="Defining-the-TensorFlow-graph">Defining the TensorFlow graph<a class="anchor-link" href="#Defining-the-TensorFlow-graph">¶</a></h3><p>Now that we have defined the SPN graph we can declare the TensorFlow operations needed for training and evaluation. The <code>MPEState</code>
#  class can be used to find the MPE state of any node in the graph. In 
# this case we might be interested in finding the most likely class based 
# on the evidence elsewhere. This corresponds to the MPE state of <code>class_indicators</code>.</p>
# <p>Note that for the gradient optimizer we use <code>AMSGrad</code>, 
# which usually yields reasonable results much faster than Adam. 
# Admittedly, more time needs to be spent on the interdependencies of 
# parameters (e.g. <code>scale_init</code>) affect training</p>
# 

# In[5]:



from libspn.examples.convspn.amsgrad import AMSGrad

# Op for initializing all weights
weight_init_op = spn.initialize_weights(root)
# Op for getting the log probability of the root
root_log_prob = root.get_log_value(inference_type=inference_type)

# Set up ops for discriminative GD learning
gd_learning = spn.GDLearning(
    root=root, learning_task_type=spn.LearningTaskType.SUPERVISED,
    learning_method=spn.LearningMethodType.DISCRIMINATIVE)
optimizer = AMSGrad(learning_rate=learning_rate)

# Use post_gradients_ops = True to also normalize weights (and clip Gaussian variance)
gd_update_op = gd_learning.learn(optimizer=optimizer, post_gradient_ops=True)

# Compute predictions and matches
mpe_state = spn.MPEState()
root_marginalized = spn.Sum(root.values[0], weights=root.weights)
marginalized_ivs = root_marginalized.generate_latent_indicators(
    feed=-tf.ones_like(class_indicators.feed)) 
predictions, = mpe_state.get_state(root_marginalized, marginalized_ivs)
with tf.name_scope("MatchPredictionsAndTarget"):
    match_op = tf.equal(tf.to_int64(predictions), tf.to_int64(class_indicators.feed))


# 
# <h3 id="Training-the-SPN">Training the SPN<a class="anchor-link" href="#Training-the-SPN">¶</a></h3>
# 

# In[6]:



# Set up some convenient iterators
train_iterator = DataIterator([train_x, train_y], batch_size=batch_size)
test_iterator = DataIterator([test_x, test_y], batch_size=batch_size)

def fd(x, y):
    return {normal_leafs: x, class_indicators: y}

with tf.Session() as sess:
    # Initialize things
    sess.run([tf.global_variables_initializer(), weight_init_op])
    
    # Do one run for test likelihoods
    matches = []
    for batch_x, batch_y in test_iterator.iter_epoch("Testing"):
        batch_matches = sess.run(match_op, fd(batch_x, batch_y))
        matches.extend(batch_matches.ravel())
        test_iterator.display_progress(Accuracy="{:.2f}".format(np.mean(batch_matches)))
    mean_test_accuracy = np.mean(matches)
    
    print("Before training test accuracy = {:.2f}".format(mean_test_accuracy))                              
    for epoch in range(num_epochs):
        
        # Train
        matches = []
        for batch_x, batch_y in train_iterator.iter_epoch("Training"):
            batch_matches, _ = sess.run(
                [match_op, gd_update_op], fd(batch_x, batch_y))
            matches.extend(batch_matches.ravel())
            train_iterator.display_progress(Accuracy="{:.2f}".format(np.mean(batch_matches)))
        mean_train_accuracy = np.mean(matches)
        
        # Test
        matches = []
        for batch_x, batch_y in test_iterator.iter_epoch("Testing"):
            batch_matches = sess.run(match_op, fd(batch_x, batch_y))
            matches.extend(batch_matches.ravel())
            test_iterator.display_progress(Accuracy="{:.2f}".format(np.mean(batch_matches)))
        mean_test_accuracy = np.mean(matches)
        
        # Report
        print("Epoch {}, train accuracy = {:.2f}, test accuracy = {:.2f}".format(
            epoch, mean_train_accuracy, mean_test_accuracy))
    


# In[ ]:




