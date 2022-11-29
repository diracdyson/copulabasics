import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

returnsant = pd.read_csv('returnsant.csv' ).drop('perf_date',axis=1 )

cov=returnsant.cov()
iidnorm= np.random.randn(returnsant.values.shape[1],1000)
SD = np.mat(np.sqrt(np.diag(cov))) # sqrt of diag vector, i.e. std dev
R = np.array(cov/np.multiply(SD, SD.T))
A= np.linalg.cholesky(R)
mvnorm= A @ iidnorm
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(mvnorm[0], mvnorm[3], c=mvnorm[1], alpha=0.1)
plt.show()

np.linalg.cholesky(returnsant[['Hedge Fund','Factor - Crowding']].corr())
