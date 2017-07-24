import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("F:/MNIST_data/", one_hot=True)


# This is a tensor for decribing our true labels for the data
y_ = tf.placeholder(tf.float32, [None, 10])

class neural_network:
    def __init__(self):
        # A 28*28 tensor which will be represent our data
        self.X = tf.placeholder(tf.float32, [None, 784])
        # There are our weights and biases
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        # The output of the neural network
        self.y = tf.add(tf.matmul(self.X,self.W),self.b)

    def train(self):
        # The loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        # The fixer
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1000):
                epoch_x, epoch_y = mnist.train.next_batch(100)     
                _, epoch_loss = sess.run([optimizer,loss], feed_dict={self.X: epoch_x, y_: epoch_y})
                print('Epoch', epoch, 'completed out of',epoch,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print('Accuracy:',accuracy.eval({self.X:mnist.test.images, y_:mnist.test.labels})*100)

def main():
    Brain = neural_network()
    Brain.train()
if __name__ == "__main__":
    main()