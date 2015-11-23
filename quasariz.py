#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(447)

# Lee datos y hace un ajuste lineal
bandI = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(80,))
errI = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(81,))
bandZ = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(82,))
errZ = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(83,))
coef = np.polyfit(bandI, bandZ, 1)

# Simulación de Monte-Carlo
N_mc = 1000
params = np.zeros((2, N_mc))
for k in range(N_mc):
    fake_bandI = np.random.normal(loc=bandI, scale=errI)
    fake_bandZ = np.random.normal(loc=bandZ, scale=errZ)
    f_param = np.polyfit(fake_bandI, fake_bandZ, 1)
    params[0][k] = f_param[0]
    params[1][k] = f_param[1]
a = np.sort(params[0])
b = np.sort(params[1])
a_025 = a[N_mc/40]
b_025 = b[N_mc/40]
a_975 = a[N_mc - N_mc/40 - 1]
b_975 = b[N_mc - N_mc/40 - 1]
plt.figure(1)
plt.errorbar(bandI, bandZ, xerr=errI, yerr=errZ, fmt='r.', label='Mediciones')
plt.plot(bandI, coef[0]*bandI + coef[1], 'b', label='Mejor ajuste')
plt.plot(bandI, a_975*bandI + b_975, 'g--', label='Regi'u'ó''n de confianza')
plt.legend(loc=2)
plt.plot(bandI, a_025*bandI + b_025, 'g--')
plt.title('Cu'u'á''sares: banda z v/s banda i')
plt.xlabel('Flujo en banda i [$10^{-6}$ Jy]')
plt.ylabel('Flujo en banda z [$10^{-6}$ Jy]')
plt.show()
