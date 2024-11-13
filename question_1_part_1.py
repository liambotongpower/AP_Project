import numpy as np
import matplotlib.pyplot as pyplot

numberOfSimulations = 10000
deltaTValue = 10000

deltaT = 1 / deltaTValue                                                            
times = np.zeros((numberOfSimulations, deltaTValue))                                          

for i in range(1, deltaTValue):                                                     
    increments = np.random.normal(0, np.sqrt(deltaT), numberOfSimulations)          
    times[:, i] = times[:, i - 1] + increments

numberOfPositives = np.sum(times > 0, axis = 1)
proportions = numberOfPositives / deltaTValue

pyplot.figure(figsize=(12, 6))
pyplot.hist(proportions, bins=30, alpha=0.5, edgecolor='black', label=f'Δt = {deltaT}')
pyplot.title('Distribution of Proportion of Positive Temperature Over Time')
pyplot.xlabel('Proportion of positive temperature')
pyplot.ylabel('Frequency')
pyplot.legend(loc='upper right')
pyplot.show()
