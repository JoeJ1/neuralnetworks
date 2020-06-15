import numpy as np
import random
import sys
import mnist # This is just to get the training data working 

class Net: # Simple neural network for classification problems
    def __init__(self,nodes,initialisation=random.random): 
        # Initialisation is the function which is used to intialise the weights and biases
        self.nodes = nodes; # Number of nodes in each row, e.g. [2,3,3,2], 
                            # for a network with 2 input neurons,
                            # two hidden layers of 3 neurons and 2 output neurons
        self.weights = []; # Array of matrices for each row of weights in the network
        self.biases = [];
        self.gamma = 0.01; # Step size
        totalWeights = 0; 
        

        # Initialise weights and biases
        for i in range(1,len(self.nodes)): # For each row (after input layer)
            self.weights.append([]);
            self.biases.append([]);
            for j in range(self.nodes[i]): # For each node in the row
                self.weights[i-1].append([]);
                self.biases[i-1].append(initialisation());
                for k in range(self.nodes[i-1]): # For each node in the previous layer
                    self.weights[i-1][j].append(initialisation());
                    totalWeights+=1;
                totalWeights+=1;
        # print(">> Total weights and biases: "+str(totalWeights));
    
    def save(self,file_name="network.npy"): # Save weights to file
        np.save(file_name,np.asarray(self.weights));
    
    def load(self,file_name="network.npy"): # Load weights from file
        self.weights = np.load(file_name,allow_pickle=True);
    
    def sigmoid(self,x): # Maybe these functions shouldn't be part of the class but I wanted to keep everything together
        return(1.0/(1.0+np.exp(-x)));
    
    def sigmoid_d(self,x): # Derivative of the sigmoid function σ'(x)=σ(x)*(1-σ(x))
        return(1.0/(1.0+np.exp(-x))*(1.0-(1.0/(1.0+np.exp(-x)))));

    def guess(self,data): # Feed forward
        current_layer = [];
        previous_layer = data;
        for i in range(1,len(self.nodes)):
            try:
                current_layer = np.array([j for j in self.weights[i-1]]).dot(previous_layer); # Sum weights and previous layer
            except IndexError:
                sys.stderr.write("Error: Training data does not match input layer");
                return(1);
            for j in range(len(current_layer)):
                current_layer[j] = current_layer[j]+self.biases[i-1][j]; # Add bias
            print(current_layer);
            current_layer = [self.sigmoid(j) for j in current_layer];
            #print(current_layer);
            previous_layer = current_layer;
        return(current_layer);
    
    def train(self,data,label): # Label is the desired output for a given training point 
        output = self.guess(data);
        error  = []; 
        try:
            # Error is difference between output and expected output (label)
            error = [(output[i]-label[i])**2 for i in range(len(label))]; 
            cost = sum(error);
        except IndexError:
            # Label must be an array of equal length to the last layer in the network
            sys.stderr.write("Error: Training data labels does not match output layer");
            return(1);
        print(error);
        print(cost);
        #for i in range(len(weights)): # Each row in network
        #    for j in range(len(weights[i])): # Each node in network
        #        for k in range(len(weights[i][j])): # Each weight and bias in the network
                
#def sigmoid(x):
#    return(1.0/(1.0+np.exp(-x)));
#
#def sigmoid_d(x): # Derivative of the sigmoid function σ'(x)=σ(x)*(1-σ(x))
#    return(1.0/(1.0+np.exp(-x))*(1.0-(1.0/(1.0+np.exp(-x)))));

def f():
    return(2);

# n = Net([784,16,16,10]);
n= Net([2,3,3,2],f);
print(n.guess([1,2]));
#n.train([1,2],[0,1]);

# mndata = mnist.MNIST('data');

# train_images,train_labels = mndata.load_training();
# test_images,test_labels = mndata.load_testing();
# index = randrange(0,len(images));
# image = np.array(images[index], dtype="uint8");
