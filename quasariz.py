#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(347)


bandI = np.loadtxt("data/DR9Q.dat", usecols=(80,))
errI = np.loadtxt("data/DR9Q.dat", usecols=(81,))
bandZ = np.loadtxt("data/DR9Q.dat", usecols=(82,))
errZ = np.loadtxt("data/DR9Q.dat", usecols=(83,))

plt.figure(1)
plt.errorbar(bandI, bandZ, xerr=errI, yerr=errZ, fmt='.')
plt.show()
