import math
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

#
# Proceso generador de datos
#
#def f(x):
    #y = [math.exp(0.15 * u + 1) if u < 5 else math.exp(0.25 * u + 1) for u in x]
    #return np.array(y)
x=np.array([438901,5266843,877786,234048,5266843,146301,234048,234048,234048,438901,877786,877786,438901,146301,234048,877786,438901,146301,234048,877786,234048,146301,146301,877786,234048,146301,234048,438901,146301,234048,234048,438901,438901,438901,877786,877786,877786,877786,438901,438901,438901,438901,438901,438901,877786,877786,877786,877786,877786,877786,146301,234048,438901,438901,438901,146301])
x

## Función de LogaritmoNatural
def LogaritmoNatural (z):
   return np.log(z)
 

## Función inversa de LogaritmoNatural
def LogaritmoNatural_Inv (z):
    return np.exp(z)
	
  
d=LogaritmoNatural(x) 
d

LogaritmoNatural_Inv (d)
