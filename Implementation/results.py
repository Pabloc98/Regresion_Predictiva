import Def.Metrics.mse_r2 as mse
import numpy as np
import matplotlib.pyplot as plt

def results(Y_test, Y_pred):
  
  mse.mse_r2(Y_test, Y_pred)
  l = len(Y_pred)
  x = np.linspace(start=0,stop=10,num=l)
  plt.scatter(x,Y_test, s=3)
  plt.scatter(x, Y_pred, s=2, c='orange')
  
