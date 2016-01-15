import csv
import numpy as np
'''
'HS,'AS','HST,'AST,'HC,'AC,'B365H,'B365D,'B365A,'BSH,'BSD,'BSA,'BWH,'BWD,'BWA,'GBH,'GBD,'GBA,'IWH,'IWD,'IWA,'LBH,'LBD,'LBA,'PSH,'PSD,'PSA,'SOH,'SOD,'SOA,'SBH,'SBD,'SBA,'SJH,'SJD,'SJA

FTR
'''

header_in = ['HS','AS','HST','AST','HC','AC','B365H','B365D','B365A']
File = ['E0.csv', 'E1.csv', 'E2.csv', 'E3.csv','E4.csv','E5.csv','E6.csv','E7.csv','E8.csv','E9.csv']
result = {'H':[1,0,0], 'D':[0,1,0],'A':[0,0,1]}

Number_inputs = 10
Number_outputs = 3
Total_games = 37 # per season for two teams

class csvreader:
    def __init__(self, Num_Inputs ,Num_Outputs, Num_games):
        self.m_inputs = Num_Inputs
        self.m_outputs = Num_Outputs
        self.m_games = Num_games
        self.m_input_data = np.ones(shape = (Num_games,Num_Inputs)) # 37 such that two teams play a total of 37 games # 10 for the total number of input neurons
#inputs = np.matrix(inputs)
        self.m_output_data = np.zeros(shape = (Num_games,Num_Outputs)) # 3 for the output neurons
#outputs = np.matrix(outputs)
        return 
    
    def extract_traindata(self, HOME, AWAY):
        # iterate for files here
        with open('E0.csv') as csvfile:
            reader  = csv.DictReader(csvfile)
            x = 0
            for row in reader:
                if row["HomeTeam"] == HOME or row["AwayTeam"] == AWAY:
                    y = 0
                    for i in header_in:
                        self.m_input_data[x][y] = row.get(i,0)
                        y += 1
                    temp = row['FTR']
                    self.m_output_data[x] = np.matrix(result[temp])
                    x += 1
        self.m_input_data = np.matrix(self.m_input_data)
        self.m_output_data = np.matrix(self.m_output_data)
        
        return

