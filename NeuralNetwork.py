from Matrix import Matrix
from math import exp
import random


def sigmoid(x): #Activation function
    return 1 / (1 + exp(-x))

def dsigmoid(y): #gradient function for sigmoid (derivative)
    #return (sigmoid(x)*(1-sigmoid(x)))
    return y * (1-y) #y has already been passed thru sigmoid


class NeuralNetwork():
    
    def __init__(self,input_nodes,hidden_nodes,output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes   
        self.lr = 0.1

        #Creation of weight matrices
        self.weights_IH = Matrix(self.hidden_nodes,self.input_nodes)
        self.weights_HO = Matrix(self.output_nodes,self.hidden_nodes) 
        self.weights_IH.uniform(-1,1)
        self.weights_HO.uniform(-1,1)

        #Creation of bias matrices
        self.bias_H = Matrix(self.hidden_nodes,1)
        self.bias_O = Matrix(self.output_nodes,1)
        self.bias_H.uniform(-1,1)
        self.bias_O.uniform(-1,1)

            
    #Moves data forward throught the NN using linear algebra
    def feedforward(self,input_list):

        #Inputs--Hidden part
        inputs = Matrix.fromList(input_list) #create input matrix object from list
        hidden = Matrix.dot_product(self.weights_IH,inputs) # weigths-inputs combined sum
        hidden.add(self.bias_H) #adding bias 
        hidden.map(sigmoid) #apply activation function to each element

        #Hidden--Output part
        output = Matrix.dot_product(self.weights_HO,hidden) #weights-hidden combined sum 
        output.add(self.bias_O) #adding bias 
        output.map(sigmoid) #apply activation function to each element
        output_list = Matrix.toList(output) #creating output list from matrix object

        return output_list

    #Trains NN with error backpropagation
    def train(self,input_list,targets_list):
        #FeedForwars Actions (same stuff as ff method)
        #Inputs--Hidden part
        inputs = Matrix.fromList(input_list) #create input matrix object from list
        hidden = Matrix.dot_product(self.weights_IH,inputs) # weigths-inputs combined sum
        hidden.add(self.bias_H) #adding bias 
        hidden.map(sigmoid) #apply activation function to each element

        #Hidden--Output part
        outputs = Matrix.dot_product(self.weights_HO,hidden) #weights-hidden combined sum 
        outputs.add(self.bias_O) #adding bias 
        outputs.map(sigmoid) #apply activation function to each element


        #Training Actions
        #Hidden--Output part
        targets = Matrix.fromList(targets_list) #transform list to matrix object
        o_errors = Matrix.substract(targets,outputs) #output errors matrix

        #gradient descent calculaiton (f'(x) = f(x)*(1-f(x)) for sigmoid) and
        #tweaking weights (dw) for outputs (HO)
        o_gradients = Matrix.smap(outputs,dsigmoid) #calculating output gradient matrix
        o_gradients.multiply(o_errors) #elementwise muliplication with o_error
        o_gradients.multiply(self.lr) #multiplying with lerning rate
        hidden_t = Matrix.transpose(hidden) #dweights tweak (dw = (lr*oE*o_gradient) x Ht)
        delta_weights_HO = Matrix.dot_product(o_gradients,hidden_t)
        self.weights_HO.add(delta_weights_HO) #change OH weights with deltas
        self.bias_O.add(o_gradients) #change HO bias with deltas (only gradient)


        #Inputs--Hidden part
        who_t = Matrix.transpose(self.weights_HO) #transposing matrix for backpropagation calculations
        h_error = Matrix.dot_product(who_t,o_errors) #hidden errors matrix
        
        #gradient descent calculaiton and tweaking weights (dw) for hidden IH)
        h_gradients = Matrix.smap(hidden,dsigmoid) #calculating hidden gradient matrix
        h_gradients.multiply(h_error) #elementwise muliplication with o_error
        h_gradients.multiply(self.lr) #multiplying with lerning rate
        inputs_t = Matrix.transpose(inputs) #dweights tweak (dw = (lr*hE*h_gradient) x It)
        delta_weights_IH = Matrix.dot_product(h_gradients,inputs_t)
        self.weights_IH.add(delta_weights_IH) #change IH weights with deltas
        self.bias_H.add(h_gradients) #change IH bias with deltas (only gradient)

    #Returns a copy of this NN
    def copy(self):        

        nn_copy = NeuralNetwork(self.input_nodes,self.hidden_nodes,self.output_nodes)
        
        nn_copy.input_nodes = self.input_nodes 
        nn_copy.hidden_nodes = self.hidden_nodes
        nn_copy.output_nodes = self.output_nodes   
        nn_copy.lr = self.lr

        #Creation of weight matrices
        nn_copy.weights_IH = self.weights_IH.copy()
        nn_copy.weights_HO = self.weights_HO.copy()

        #Ceation of bias matrices
        nn_copy.bias_H = self.bias_H.copy()
        nn_copy.bias_O = self.bias_O.copy()

        return nn_copy

    #Mutates/tweaks the weights of a NN by a certain rate(%)
    def mutate(self,rate):
        def mutate(n): #mutation function (not method)
            if random.uniform(0,1) < rate:
                return random.uniform(-1,1)
            else:
                return n

        self.weights_IH.map(mutate)
        self.weights_HO.map(mutate)
        self.bias_H.map(mutate)
        self.bias_H.map(mutate)