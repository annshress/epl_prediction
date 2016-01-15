import numpy as np
from random import random
import math
import time

import xlread

Home = "Arsenal"
Away = "Chelsea"

Num_of_inputs = 10 #  plus 1 bias input
Num_patterns = 37 # 10 files X 37 patterns in each -> patterns
Num_of_outputs = 3 # H,D,A
Num_hidden = 8
Num_epoch = 5000
etaOH = 0.08 # learning rate # if you get errors try playing with these ETA values :/ i dont know why :/ :{ try not to cry and cry a lot
etaIH = 0.6

#Train_inputs = np.matrix('0 0 1 ; 0 1 1 ; 1 0 1 ; 1 1 1')
#Train_outputs = np.matrix('0 0 ; 1 0 ; 1 0 ;0 1') #outputs : XOR and AND

class BackProp:
    def __init__(self, num_inputs, num_outputs, num_patterns,num_hidden, num_epoch, train_in, train_out):
        self.m_inputs = num_inputs 
        self.m_outputs = num_outputs
        self.m_hidden = num_hidden
        self.m_patterns = num_patterns
        self.m_epoch = num_epoch
        self.m_train_in = train_in
        self.m_train_out = train_out
        self.weightsIH = np.zeros(shape = (num_hidden, num_inputs)) #input to hidden value weights
        self.weightsHO = np.zeros(shape = (num_outputs, num_hidden))
        self.activationH = np.zeros(shape = (num_hidden, 1)) # output from hidden nodes  => a
        self.w_inputH = np.zeros(shape = (num_hidden,1)) # weighted inputs into hidden
        self.w_inputO = np.zeros(shape = (num_outputs,1)) # w_inputs into output layer
        self.m_out = np.zeros(shape = (num_outputs, 1)) # output value from the network // activation value from output
        self.error = np.zeros(shape = (num_outputs, 1)) # aka gradient of the cost
        self.errorH = np.zeros(shape = (num_hidden, 1))
        self.MS_error = 0.0
        
        return

    def hadamard(self, a, b):
        return a*b
        '''if (a.shape == b.shape) :
            return a*b
        print "error in the dimensions; must be same dimensions"
        return None '''
    
    def sigmoid(self, value):
        return 1/(1 + math.exp(0-value))
        #return math.tanh(value)
    
    def _sigmoid(self,value):
        return (self.sigmoid(value) * (1 - self.sigmoid(value)))
        #return (1 - math.pow(sigmoid, 2))
        
    def assign_weight(self):
        #assign random weights ranging from..
        try: # if weights already stored in a file
            weights = np.load('weights.npz')
            self.weightsHO = weights['arr_0']
            self.weightsIH = weights['arr_1']
        except: # if program ran for the first time
            for i in xrange(self.m_hidden):
                    for j in xrange(self.m_inputs):
                        self.weightsIH[i][j] = (random() - 0.5)/3.0
                    for k in xrange(self.m_outputs):
                        self.weightsHO[k][i] = (random() - 0.5)/2.0
                
        return

    def create_matrices(self):
        self.weightsIH = np.matrix(self.weightsIH)
        self.weightsHO = np.matrix(self.weightsHO)
        self.activationH =  np.matrix(self.activationH) # output from hidden nodes  => a
        self.w_inputH = np.matrix(self.w_inputH) # z values
        self.w_inputO =  np.matrix(self.w_inputO) 
        self.m_out = np.matrix(self.m_out) # output value from the network
        self.error = np.matrix(self.error)
        self.errorH = np.matrix(self.errorH)
    
    def feed_forward(self, index):
        vsig = np.vectorize(self.sigmoid)
        
        self.w_inputH = self.weightsIH.dot(self.m_train_in[index].getT())
        self.activationH = vsig(self.w_inputH)
        # now fr the H->O layer
        self.w_inputO = self.weightsHO.dot(self.activationH)
        self.m_out = vsig(self.w_inputO)

        return
    
    def output_error(self, index):
        vhada = np.vectorize(self.hadamard)
        _vsig = np.vectorize(self._sigmoid)
        self.error = vhada((self.m_out - self.m_train_out[index].getT()),_vsig(self.w_inputO))

        return

    def hidden_error(self):
        vhada = np.vectorize(self.hadamard)
        _vsig = np.vectorize(self._sigmoid)
        self.errorH = vhada(self.weightsHO.getT().dot(self.error), _vsig(self.w_inputH))
        
        return
    
    def update_weights(self, index):
        #OH weights
        self.weightsHO = self.weightsHO - etaOH*(self.error.dot(self.activationH.getT())) / self.m_patterns

        #IH weights
        self.weightsIH = self.weightsIH - etaIH*(self.errorH.dot(self.m_train_in[index])) / self.m_patterns
        
        return
    
    def save_weights(self):
        np.savez('weights.npz', self.weightsHO, self.weightsIH)
        
        return
    
    def calc_error(self):
        
        return
    
    def train_network(self):
        #choose a random input among the training inputs
        for j in xrange(self.m_epoch):
            for train_index in xrange(self.m_patterns):
                # calculate the weighted input and the output values
                self.feed_forward(train_index)
                #calculate the outputerror
                self.output_error(train_index)
                #calculate hidden error
                self.hidden_error()
                #now update the weights
                self.update_weights(train_index)
                
            #self.MS_error = self.calc_error()
            #print ("RMS error : ",self.MS_error)
        self.save_weights()
            
        return
        
    def display_results(self):
        for i in range(self.m_patterns):
            self.feed_forward(i)
            print ("true output : " + str(self.m_train_out[i]) + " obtained:  " + str(self.m_out))
        return
   
     
if __name__ == "__main__":   
    read = xlread.csvreader(Num_of_inputs, Num_of_outputs, Num_patterns)
    read.extract_traindata(Home, Away)

    bp = BackProp(Num_of_inputs, Num_of_outputs, Num_patterns,Num_hidden, Num_epoch, read.m_input_data, read.m_output_data)
    bp.assign_weight()
    bp.create_matrices()
    start = time.time()
    #bp.display_weights()
    bp.train_network()
    #bp.display_weights()
    print str(time.time() - start) + "training time"
    bp.display_results()
    print str(time.time() - start) + "training time"

