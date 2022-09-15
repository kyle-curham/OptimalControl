# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:00:51 2022

@author: curha
"""
from OptConMethod import PGM, sysID
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

numstates = 10 # number of hidden states
numinputs = 8  # number of inputs
numchans = 60  # number of EEG electrodes
srate = 1e3    # sample frequency of the simulation
dt = 1/srate

rng = np.random.default_rng(0)

A = rng.random([numstates,numstates]) # state gain
A = np.round((A + np.transpose(A))/2)
A = A - np.diag(np.diagonal(A))

D = np.diag(1/np.sqrt(np.sum(A,axis=0)))
A = D@(A+np.eye(numstates))@D

eta = .001            # step size
epsilon = .01         # deviation from the sphere
convergence = 0.001 # convergence criteria

maxiter = 100         # max number of allowed iterations
R = np.sqrt(np.trace(np.transpose(A)@A)) # radius of the sphere

opt = PGM(eta=eta,epsilon=epsilon,convergence=convergence,maxiter=10000,radius=R)
opt.initialize(A,numinputs)
B,G,c = opt.optimize()


C = rng.random([numchans,numstates])
D = rng.random([numchans,numinputs])
ss = signal.StateSpace(A,B,C,D)
ssd = ss.to_discrete(dt=dt)

Ad = ssd.A
Bd = ssd.B
Cd = ssd.C
Dd = ssd.D
u = rng.random([int(10*srate),numinputs])/100

t,y,x = signal.dlsim(ssd, u)
plt.plot(y)

# define number of markov parmaeters
r = 20

sysid = sysID(np.transpose(y), np.transpose(u), r)
H=sysid.OKID()
cut=2*r
Ae,Be,Ce,De = sysid.ERA(H,cut)

psi,psiinv,Aest,Best,Cest = sysid.map(Ae,Be,Ce,C)