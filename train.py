import tensorflow as tf
import numpy as np


# Network config
input_size = 768
layer_size = 2048
number_of_layer = 3
output_size = 1
learning_rate = 0.001
training_epochs = 10
number_of_games = 10
batch_size = 100 # (?)

# Display log every _display_step_ step
display_step = 1


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1'], biases['b1']))
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2'], biases['b2']))
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3'], biases['b3']))
    layer_3 = tf.nn.relu(layer_3)

    out_layer = tf.matmul((layer3, weight['out']) + biases['out'])

    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([input_size, layer_size])),
    'h2': tf.Variable(tf.random_normal([layer_size, layer_size])),
    'h3': tf.Variable(tf.random_normal([layer_size, layer_size])),
    'out': tf.Variable(tf.random_normal([layer_size, output_size]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([layer_size])),
    'b2': tf.Variable(tf.random_normal([layer_size])),
    'b3': tf.Variable(tf.random_normal([layer_size])),
    'out': tf.Variable(tf.random_normal([output_size]))
}

# Construct model
model = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch training
with tf.Session() as session:
    session.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(number_of_games/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            # Get next batch
            # batch_x, batch_y = 
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            # Compute average loss
            avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch :", epoch, " ## cost :", cost

    print "Trainging completed !"

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy :", accuracy


                                                          




