import csv
import time
import math
import operator
import numpy as np
file_list = ['E0.csv','E1.csv','E2.csv','E3.csv','E4.csv','E5.csv','E6.csv','E7.csv','E8.csv','E9.csv','E10.csv']
test_file = 'E11.csv'

def minimax(ll):
    ll = [(ll[i])[:len(ll[i])-1] for i in range(len(ll))] 
    global maximum
    global minimum
    a=np.matrix(ll)

    maximum = np.max(a, axis=0) #list containing maximum values of each columns
    minimum = np.min(a, axis=0)
    
    return 

def exact_output(HOME, AWAY):
    with open("football archive/"+test_file) as csvfile:
        reader=csv.DictReader(csvfile)
        for row in reader:
            if row["HomeTeam"]==HOME and row["AwayTeam"]==AWAY:
                result=row.get("FTR","@#$%")
    print "The true resu    lt is " + result
    return result

def trainSet(HOME, AWAY):
    attributes = ['HS', 'AS', 'AST', 'HC', 'AC', 'B365H', 'B365H', 'B365D', 'B365A']
    matrix = [[1 for x in range(len(attributes)+1)] for y in range(407)] # 12 is number of file to be editted later and +1 for extra exact result
    x=0
    for file in file_list:
        with open("football archive/"+file) as csvfile:
            reader=csv.DictReader(csvfile)
            for row in reader:
                if(row["HomeTeam"]==HOME or row["AwayTeam"]==AWAY):
                    y=0
                    for i in attributes:
                        matrix[x][y] = float(row.get(i, 0))
                        y+=1
                    matrix[x][len(matrix[x])-1] = row.get("FTR")
                    x+=1
    
    minimax (matrix)
    return matrix

def testInstance(HOME, AWAY):
    attributes = ['HS', 'AS', 'AST', 'HC', 'AC', 'B365H', 'B365H', 'B365D', 'B365A']
    with open("football archive/E11.csv") as csvfile:
        matrix=[0 for x in range(len(attributes)+1)] #+1 for additional FTR result column
        reader=csv.DictReader(csvfile)
        for row in reader:
            if row["HomeTeam"]==HOME and row["AwayTeam"]==AWAY:
                y=0 
                for i in attributes:
                    matrix[y]=float(row.get(i, 0))
                    y+=1
                matrix[len(matrix)-1] = row.get("FTR")
    tstmatrix = matrix  
    return tstmatrix

def euclideanDistance(data1, data2):
    distance=0
    for x in range(len(data1)):
        distance+=pow((data1[x]-data2[x])/(maximum[0,x] - minimum[0,x]), 1)  
    return distance
    #return math.sqrt(distance)

def getNeighbors(trainSet, testInstance, k):
    distances=[]
    length=len(testInstance)
    
    for x in range(10): #length of trainSet == 10 but how the fuck we can get?
        dist=euclideanDistance(testInstance[:-1], (trainSet[x])[:-1])
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    print neighbors
    return neighbors

def getVote(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    #print sortedVotes
    print sortedVotes

def main():
    home = 'Swansea'
    away = 'Stoke'
   
    eOutput=exact_output(home,away)
    tSet=trainSet(home,away)
    tInstance=testInstance(home,away)
    nei = getNeighbors(tSet, tInstance, 10) #value of k=3
    getVote(nei)
    time.sleep(3)
    print eOutput
main()
