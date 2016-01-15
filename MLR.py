import numpy as np
from numpy.linalg import inv
from source5 import MLR
import math

class Result():
    def __init__(self):
        self.beta = np.zeros(shape = (14,1)) # 13 is the number of attributes + 1 for the beta_constant value (residual value)
        return
    
    def find_beta(self): # calculate the beta_matrix
        self.x = np.matrix(self.x)
        self.y = np.matrix(self.y)
        # beta = (Xt X)-1 Xt Y
        self.beta = (inv(self.x.getT().dot(self.x))).dot(self.x.getT()).dot(self.y)
        return
    
    def result(self, home, away): #calculate the result for current season here
        
        new = MLR()
        new.calculate_points(home, away)
        new.count_h2h(home, away)
        new.current_standings(home, away)
        new.current_season_games(home, away)
        new.last_4_games_this_season(home,away)
        new.X()
        
        self.X = new.x[0]
        self.X = np.matrix(self.X)
        # check if x-values lie between the limits of the training set
        for i in xrange(13):
            if self.X[0,i] > self.x_max[i]:
                self.X[0,i] = self.x_max[i]
            if self.X[0,i] < self.x_min[i]:
                self.X[0,i] = self.x_min[i]
        self.X = np.concatenate((self.X,np.matrix('1')),axis = 1)
        return

    def calculate_x_y(self):
        try:
            variables = np.load('Games/' + 'MLRxy.npz')
            self.x = variables['arr_0'] # init
            self.y = variables['arr_1']
            self.x_max = self.x.max(axis = 0) 
            self.x_min = self.x.min(axis = 0)
        except:
            obj = MLR()
            obj.Extract() # bring the independent variables (X)
            obj.Y() # bring in the dependent variable (y)
            #total_attr = 13 # add all the 2nd dimensions of each attributes
            obj.X() # combine all the independent variables to one
            self.x = obj.x # init
            self.y = obj.y # init
            self.x_max = self.x.max(axis = 0)
            self.x_min = self.x.min(axis = 0)
            
            obj.save()
            
        temp = np.ones(shape = (40,1))
        self.x = np.concatenate((self.x, temp),axis = 1) # 40 by 13+1 where last column has all value as ones... to obtain a constant-beta value
        
        return
    
    def calc_prob(self, value): # from the obtained y calculate the probability of the game ouptups # 
        x = math.fabs(value) # x = draw distance
        y = math.fabs(1 - value) # home distance
        z = math.fabs(-1 - value) # away distance
        sum = x+y+z
        x = x/sum
        y = y/sum
        z = z/sum
        x = 1-x
        y = 1-y
        z = 1-z
        sum = x+y+z
        x = x/sum
        y = y/sum
        z = z/sum
        return [x*100,y*100,z*100]
        
def main(home, away):
    b = Result()
    
    b.calculate_x_y()
    b.find_beta()
    #print b.x.dot(b.beta) - b.y
    b.result(home, away)
    pred = b.calc_prob(b.X.dot(b.beta))
    return pred
