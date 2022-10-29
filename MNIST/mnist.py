import numpy
import scipy.special
import matplotlib.pyplot
#% matplotlib inline


class nueralnet:
    # initilisation of nueral net instance
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # learning rate of the nueral net
        self.lr = learning_rate

        # our activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        # linked weights in the form w_i_j, with link from node i to node j. w_1_1, w_2_1, w_3_1 ; w_1_2, w_2_2....
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    # train the nueral net
    def train(self, inputs_list, targets_list):
        # step 1 is to obtain output from any training example
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # step 2 is to compare the calculated output with the target, and calculate an error which
        # will be used to update the weights

        # turn targets list into a 2d numpy array
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate the difference between each element of the array and the target value
        error_output = targets - final_outputs

        # backpropagating the error using a dot product martrix, the error is split among the links using the weights,
        # and recombined at the hidden nodes
        error_hidden = numpy.dot(self.who.T, error_output)

        # update the weights between the hidden and final output layers
        self.who += self.lr * numpy.dot(
            (error_output * self.activation_function(final_outputs) * self.activation_function((1 - final_outputs))),
            numpy.transpose(hidden_outputs))

        # update the weights between the input and hidden layers
        self.wih += self.lr * numpy.dot(
            (error_hidden * self.activation_function(hidden_outputs) * self.activation_function((1 - hidden_outputs))),
            numpy.transpose(inputs))

    # query the nueral net for output after training
    def query(self, inputs_list):
        # convert inputs to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate inputs into the hidden nodes by calculating a dot product of matrix of weights between input and hidden layers with inputs
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate outputs by using the sigmoid function
        hidden_outputs = self.activation_function(hidden_inputs)

        # same process to get inputs and outputs to the output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.2

nn = nueralnet(input_nodes, hidden_nodes, output_nodes, learning_rate)

# loading the CSV file
training_data = open('mnist_dataset/mnist_train_100.csv', 'r')
training_data_list = training_data.readlines()
training_data.close()

# training the nueral network
epochs = 4

for e in range(epochs):
    for record in training_data_list:
        # split into pixels and label list
        label_with_pixels = record.split(',')

        # shift the inputs to fit out activation function
        inputs = (numpy.asfarray(label_with_pixels[1:]) / 255.0 * 0.99) + 0.01

        # setting up targets for each
        targets = numpy.zeros(output_nodes) + 0.01

        # set the correct label index to 0.99
        targets[int(label_with_pixels[0])] = 0.99
        nn.train(inputs, targets)

# testing with MNIST testing data

# obtain testing data
testing_data = open('mnist_dataset/mnist_test_10.csv', 'r')
testing_file = testing_data.readlines()
testing_data.close()

scorecard = []  # score the nueral net

for record in testing_file:
    label_with_pixels = record.split(',')
    label = int(label_with_pixels[0])  # obtain data label
    image_inputs = numpy.asfarray(label_with_pixels[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_inputs, cmap='Greys', interpolation='None')
    inputs = (numpy.asfarray(
        label_with_pixels[1:]) / 255.0 * 0.99) + 0.01  # obtain input pixels and tweak them to fit the nueral network
    outputs = nn.query(inputs)

    net_result = numpy.argmax(outputs)  # maximum(closest to target of 0.99) is network best guess
    if net_result == label:
        scorecard.append(1)
    else:
        scorecard.append(0)

    print(f'Correct: {label} Network: {net_result}')

score_array = numpy.asfarray(scorecard)

print(f'PERFORMANCE: {(score_array.sum() / score_array.size) * 100}%')


