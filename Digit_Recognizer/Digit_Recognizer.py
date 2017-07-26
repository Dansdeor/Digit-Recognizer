import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("F:/MNIST_data/", one_hot=True)


# This is a tensor for describing our true labels for the data
y_ = tf.placeholder(tf.float32, [None, 10])

class deep_neural_network:
    def __init__(self):
        # A 28*28 tensor which will be represent our data
        self.X = tf.placeholder(tf.float32, [None, 784])
        # There are our weights and biases
        
        # I decided to build a deep neural network of 500 neuron for each layer
        W = tf.Variable(tf.random_normal([784, 500]))
        b = tf.Variable(tf.random_normal([500]))
        self.l1 = layer(self.X,W,b)
       
        W = tf.Variable(tf.random_normal([500, 500]))
        b = tf.Variable(tf.random_normal([500]))
        self.l2 = layer(self.l1,W,b)
       
        #This is the output part of the network
        W = tf.Variable(tf.random_normal([500, 10]))
        b = tf.Variable(tf.random_normal([10]))
        # The output of the neural network
        self.y = tf.add(tf.matmul(self.l2,W),b)

    def train(self):
        # The loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        # The fixer
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                for _ in range(int(mnist.train.num_examples/100)):
                    epoch_x, epoch_y = mnist.train.next_batch(100)     
                    _, epoch_loss = sess.run([optimizer,loss], feed_dict={self.X: epoch_x, y_: epoch_y})
                    print('Epoch', epoch, 'completed out of',10,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print('Accuracy:',accuracy.eval({self.X:mnist.test.images, y_:mnist.test.labels})*100)

def layer(X,W,b):
    layer = tf.add(tf.matmul(X,W),b)
    layer = tf.nn.relu(layer)
    return layer

def main():
    Brain = deep_neural_network()
    Brain.train()
if __name__ == "__main__":
    main()