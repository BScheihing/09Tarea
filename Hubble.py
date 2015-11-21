#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(347)

def fit_data(d,v):
    fit1 = np.sum(d*v)/np.sum(d*d)
    fit2 = np.sum(v*v)/np.sum(d*v)
    H0 = (fit1 + fit2) / 2.0
    return H0

datosHubble = np.loadtxt("data/hubble_original.dat")
datosSNI = np.loadtxt("data/SNIa.dat", usecols=(1,2))
dHubble = np.zeros(len(datosHubble))
vHubble = np.zeros(len(datosHubble))
dSNI = np.zeros(len(datosSNI))
vSNI = np.zeros(len(datosSNI))
for i in range(len(datosHubble)):
    dHubble[i] = datosHubble[i][0]
    vHubble[i] = datosHubble[i][1]
for i in range(len(datosSNI)):
    dSNI[i] = datosSNI[i][1]
    vSNI[i] = datosSNI[i][0]

H = fit_data(dHubble, vHubble)
H2 = fit_data(dSNI, vSNI)
plt.figure(1)
plt.plot(dHubble, vHubble, 'o')
plt.plot(dHubble, H*dHubble)
plt.figure(2)
plt.plot(dSNI, vSNI, 'o')
plt.plot(dSNI, H2*dSNI)
plt.show()
