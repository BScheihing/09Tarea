#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(547)


def fit_data(I, Z):
    coef1 = np.polyfit(I, Z, 1)
    coef2 = np.polyfit(Z, I, 1)
    x_c = (coef1[1]*coef2[0]+coef2[1])/(1.0-coef1[0]*coef2[0])
    y_c = coef1[0]*x_c + coef1[1]
    m = np.tan((np.arctan(coef1[0]) + np.arctan(1.0/coef2[0])) / 2.0)
    return (m, y_c - m*x_c)


# Lee datos y hace un ajuste lineal
bandI = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(80,))
errI = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(81,))
bandZ = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(82,))
errZ = 3.631*np.loadtxt("data/DR9Q.dat", usecols=(83,))
coef = fit_data(bandI, bandZ)

# Simulación de Monte-Carlo
N_mc = 100000
params = np.zeros((2, N_mc))
for k in range(N_mc):
    fake_bandI = np.random.normal(loc=bandI, scale=errI)
    fake_bandZ = np.random.normal(loc=bandZ, scale=errZ)
    f_param = fit_data(fake_bandI, fake_bandZ)
    params[0][k] = f_param[0]
    params[1][k] = f_param[1]
a = np.sort(params[0])
b = np.sort(params[1])
a_025 = a[N_mc/40]
b_025 = b[N_mc/40]
a_975 = a[N_mc - N_mc/40 - 1]
b_975 = b[N_mc - N_mc/40 - 1]
print "Ajuste lineal entre flujo de banda i y banda z"
print "fz = a * fi + b"
print "a =", coef[0]
print "Intervalo de confianza para a: ", (a_025, a_975)
print "b =", coef[1], "[10^-6 Jy]"
print "Intervalo de confianza para b: ", (b_025, b_975), "[10^-6 Jy]"
xI = np.linspace(-20, 500, 500)
plt.figure(1)
plt.errorbar(bandI, bandZ, xerr=errI, yerr=errZ, fmt='r.', label='Mediciones')
plt.plot(xI, coef[0]*xI + coef[1], 'b', label='Mejor ajuste')
plt.plot(xI, a_975*xI + b_975, 'g--', label='Regi'u'ó''n de confianza')
plt.legend(loc=2)
plt.plot(xI, a_025*xI + b_025, 'g--')
plt.title('Cu'u'á''sares: banda z v/s banda i')
plt.xlabel('Flujo en banda i [$10^{-6}$ Jy]')
plt.ylabel('Flujo en banda z [$10^{-6}$ Jy]')
plt.xlim([-20, 500])
plt.savefig('bandsiz.eps')
plt.show()
