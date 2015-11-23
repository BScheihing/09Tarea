#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(347)


def fit_data(d, v):
    fit1 = np.sum(d*v)/np.sum(d*d)
    fit2 = np.sum(v*v)/np.sum(d*v)
    H0 = np.tan((np.arctan(fit1) + np.arctan(fit2)) / 2.0)
    return H0


def bootstrap(d, v, N_B):
    H_gen = np.zeros(N_B)
    d_boot = np.zeros(len(d))
    v_boot = np.zeros(len(v))
    for k in range(N_B):
        indices = np.random.randint(low=0, high=len(d), size=len(d))
        for i in range(len(d)):
            d_boot[i] = d[indices[i]]
            v_boot[i] = v[indices[i]]
        H_gen[k] = fit_data(d_boot, v_boot)
    H_gen = np.sort(H_gen)
    H_025 = H_gen[N_B/40]
    H_975 = H_gen[N_B - N_B/40 - 1]
    return (H_025, H_975)

datosHubble = np.loadtxt("data/hubble_original.dat")
datosSNI = np.loadtxt("data/SNIa.dat", usecols=(1, 2))
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

N1 = len(dHubble)
H1 = fit_data(dHubble, vHubble)
H1_conf = bootstrap(dHubble, vHubble, int(N1**2))
N2 = len(dSNI)
H2 = fit_data(dSNI, vSNI)
H2_conf = bootstrap(dSNI, vSNI, int(N2**2))

plt.figure(1)
plt.plot(dHubble, vHubble, 'ro', label='Mediciones originales')
plt.plot(dHubble, H1*dHubble, 'b', label='Mejor ajuste')
plt.plot(dHubble, H1_conf[0]*dHubble, 'g--', label='Regi'u'ó''n de confianza')
plt.legend(loc=2)
plt.plot(dHubble, H1_conf[1]*dHubble, 'g--')
plt.title('Datos originales de Edwin Hubble')
plt.xlabel('Distancia D a las nebulosas [Mpc]')
plt.ylabel('Velocidad de recesi'u'ó''n de las nebulosas [km/s]')
plt.figure(2)
plt.plot(dSNI, vSNI, 'ro', label='Mediciones')
plt.plot(dSNI, H2*dSNI, 'b', label='Mejor ajuste')
plt.plot(dSNI, H2_conf[0]*dSNI, 'g--', label='Regi'u'ó''n de confianza')
plt.legend(loc=2)
plt.plot(dSNI, H2_conf[1]*dSNI, 'g--')
plt.title('Datos usando supernovas tipo I')
plt.xlabel('Distancia D a las nebulosas [Mpc]')
plt.ylabel('Velocidad de recesi'u'ó''n de las nebulosas [km/s]')
plt.show()
