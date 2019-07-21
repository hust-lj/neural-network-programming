import numpy
import scipy.special

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes, outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x:scipy.special.expit(x)
    def train(self,input_list,target_list):
        inputs = numpy.array(input_list,ndmin=2).T
        targets = numpy.array(target_list,ndmin=2).T
        hidden_input = numpy.dot(self.wih,inputs)
        hidden_out = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who,hidden_out)
        final_out = self.activation_function(final_input)
        output_error = targets - final_out
        hiden_error = numpy.dot(self.who.T,output_error)
        self.who += self.lr * numpy.dot((output_error*final_out*(1-final_out)),numpy.transpose(hidden_out))
        self.wih += self.lr * numpy.dot((hiden_error * hidden_out * (1-hidden_out)) , numpy.transpose(inputs))

    def query(self,input_list):
        inputs = numpy.array(input_list,ndmin=2).T
        hidden_input = numpy.dot(self.wih,inputs)
        hidden_out = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who,hidden_out)
        final_out = self.activation_function(final_input)
        return final_out

with open('mnist_train.csv') as f:
    train_data_list = f.readlines()
# all_values=data_list[0].split(',')
# scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# print(scaled_input)
n=neuralNetwork(784,200,10,0.1)
for i in range(4):
    for line in train_data_list:
        all_values = line.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(10)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

with open('mnist_test.csv') as f:
    test_data_list = f.readlines()
right_list = []
for line in test_data_list:
    all_values = line.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    out = n.query(inputs)
    label = numpy.argmax(out)
    if int(all_values[0]) == label:
        right_list.append(1)
    else:
        right_list.append(0)

print float( right_list.count(1))/float(len(right_list))