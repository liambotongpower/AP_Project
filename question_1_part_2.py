import numpy as np
import matplotlib.pyplot as pyplot

numberOfSimulations = 10000
deltaTValue = 100

deltaT = 1 / deltaTValue                                                            
times = np.zeros((numberOfSimulations, deltaTValue))                                          

for i in range(1, deltaTValue):                                                     
    increments = np.random.normal(0, np.sqrt(deltaT), numberOfSimulations)          
    times[:, i] = times[:, i - 1] + increments

maxIndices = np.argmax(times, axis=1)
maxTimes = maxIndices * deltaT

pyplot.hist(maxTimes, bins=30, alpha=0.5, edgecolor='black', label=f'Δt = {deltaT}')
pyplot.title('Distribution of Time at Maximum Temperature')
pyplot.xlabel('Time of maximum temperature')
pyplot.ylabel('Frequency')
pyplot.legend(loc='upper right')
pyplot.show()