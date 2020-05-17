#!/usr/bin/python

import os
import sys
import numpy as np
from scipy.optimize import minimize

####################
## Subroutines    ##
####################
def computeAffinity(Avec, K, T, expr, nCons):
    nSites = T.shape[0]    
    A = np.zeros_like(T)
    (x, y) = np.where(T != 0)
    
    for i in range(0, len(x)):
        A[x[i], y[i]] = Avec[i]
    
    J = 0
    for j in range(0, nCons):
        for i in range(0, nSites):
            a_ex = K[:, j] * A[i, :]
            avg = sum(a_ex) / sum(A[i, :])
            if ~np.isnan(avg):
                J = J + (avg - expr[i, j]) ** 2    

    return (J)


def computeAffinityGrad(Avec, K, T, expr, nCons):
    nSites = T.shape[0]
    nKins = T.shape[1]
    A = np.zeros_like(T)
    (x, y) = np.where(T != 0)
    nPairs = len(x)
    
    for i in range(0, nPairs):
        A[x[i], y[i]] = Avec[i]
    
    grad = np.zeros((nPairs))
    for i in range(0, nSites):
        for j in range(0, nKins):
            g = 0
            for l in range(0, nCons):
                if A[i, j] != 0:
                    g = g + (2 * ((K[j, l] * A[i, j]) / sum(A[i, :]) - expr[i, l]) *
                             ((K[j, l] * sum(A[i, :]) - sum(K[:, l] * A[i, :])) / (sum(A[i, :]) ** 2))) 
            if A[i, j] != 0:
                grad[i] = g    

    return(grad)

def computeActivity(K, A, T, expr):
    Avec = A
    nSites = T.shape[0]    
    A = np.zeros_like(T)
    J = 0
    (x, y) = np.where(T != 0)
    
    for i in range(0, len(x)):
        A[x[i], y[i]] = Avec[i]
    
    for i in range(0, nSites):
        if ~np.isnan(expr[i]):
            a_ex = K * A[i, :]
            avg = sum(a_ex) / sum(A[i, :])
            if ~np.isnan(avg):
                J = J + (avg - expr[i]) ** 2

    return (J)

def computeActivityGrad(K, A, T, expr):
    Avec = A
    nSites = T.shape[0]
    nKins = T.shape[1]
    A = np.zeros_like(T)
    (x, y) = np.where(T != 0)
    nPairs = len(x)    
    
    for i in range(0, nPairs):
        A[x[i], y[i]] = Avec[i]
    
    grad = np.zeros(nKins)
    for l in range(0, nKins):
        N = 0
        for i in range(0, nSites):
            if ~np.isnan(expr[i]) and T[i, l] != 0:
                a_ex = K * A[i, :]
                avg = sum(a_ex) / sum(A[i, :])
                if ~np.isnan(avg):
                    N = N + 2 * (avg - expr[i]) * (A[i, l] /sum(A[i, :]))
        grad[l] = N    

    return(grad)

def fitActivities(T, expr, nCons, nIters):
    ## Fit kinase activities and affinities
    nKins = T.shape[1]
    nPairs = int(sum(sum(T)))
    
    ## For affinity, A
    minA = 0.1
    maxA = 10
    bndsA = ((minA, maxA), ) * nPairs
    
    ## For activity, K
    minK = -15
    maxK = 15
    bndsK = ((minK, maxK), ) * nKins
    
    ## Multiple results over nIters
    As = np.zeros((nPairs, nIters))
    Ks = np.zeros((nIters, nKins, nCons))
    
    for i in range(0, nIters):
        print ("Iteration #%d\n" % (i + 1))
        J1 = 10000
        J2 = 1000
        A = np.random.uniform(minA, maxA, nPairs)
        K = np.random.uniform(minK, maxK, (nKins, nCons))
        
        while (J1 - J2 > 10):
            ## Minimization of a global cost function
            res = minimize(computeAffinity, A, method = "SLSQP", jac = computeAffinityGrad, 
                           args = (K, T, expr, nCons), bounds = bndsA, options = {'disp': True})
            J1 = res.fun
            A = res.x
            As[:, i] = A
            
            J2 = 0
            for j in range(0, nCons):
                ## Minimization of a local cost function for each condition (sample)
                res = minimize(computeActivity, K[:, j], method = "SLSQP", jac = computeActivityGrad,
                                args = (A, T, expr[:, j]), bounds = bndsK, options = {'disp': True})
                K[:, j] = res.x
                J2 = J2 + res.fun

            Ks[i, :, :] = K
    
    J = np.zeros((nIters))
    for i in range(0, nIters):
        J[i] = computeAffinity(As[:, i], Ks[i, :, :], T, expr, nCons)
    
    minInd = np.argmin(J)
    A = As[:, minInd]
    K = Ks[minInd, :, :]
    return (A, K)

#####################
## Main routine    ##
#####################
## Data loading
inputFile = sys.argv[1]
data = np.genfromtxt(inputFile, skip_header = 1, delimiter = "\t", dtype = str)
sites = data[:, 0]
siteKins = data[:, 1]
expr = data[:, 2:]
expr = expr.astype(np.float)
nCons = expr.shape[1]

## Make a list of kinases
kins = []

for i in range(0, len(siteKins)):
    iKins = (siteKins[i].replace('"', '')).split(",")
    for iKin in iKins:
        kins.append(iKin)

kins = sorted(set(kins))

## Create a truth table (T = [nSites x nKins])
nSites = len(sites)
nKins = len(kins)
T = np.zeros((nSites, nKins))

for i in range(0, nSites):
    iKins = (siteKins[i].replace('"', '')).split(",")
    for j in range(0, len(iKins)):
        for k in range(0, len(kins)):
            if iKins[j] == kins[k]:
                T[i, k] = 1

## Fit activities (and affinities too)
## For each trial, "nIters" times of optimizations will be performed and 
## the result producing the minimum objective function will be selected
nPairs = int(sum(sum(T)))
nRuns = 3
As = np.zeros((nPairs, nRuns))
Ks = np.zeros((nKins, nCons * nRuns))
for i in range(0, nRuns):
    print ("\nRun %d is being performed\n" % (i + 1))
    nIters = 3
    (iA, iK) = fitActivities(T, expr, nCons, nIters)
    As[:, i] = iA
    Ks[:, i * nCons : (i + 1) * nCons] = iK

## Save output variables to files
base = os.path.basename(inputFile)
base = os.path.splitext(base)[0]
outputFileA = base + "_ikap_affinitiy.txt"
headerA = ["Sites", "Kinases"]
for i in range(1, nRuns + 1):
    colName = "Run_" + str(i)
    headerA.append(colName)

(x, y) = np.where(T != 0)
sitesA = []
kinsA = []
for i in range(0, len(x)):
    sitesA.append(sites[x[i]])
    kinsA.append(kins[y[i]])

outputA = np.c_[sitesA, kinsA, As]
outputA = np.vstack((headerA, outputA))
np.savetxt(outputFileA, outputA, delimiter = "\t", fmt = "%s")

outputFileK = base + "_ikap_activity.txt"
headerK = ["Kinases"]
for i in range(1, nRuns + 1):
    for j in range(1, nCons + 1):
        colName = "Run_" + str(i) + "_Data_" + str(j)
        headerK.append(colName)
outputK = np.c_[kins, Ks]
outputK = np.vstack((headerK, outputK))
np.savetxt(outputFileK, outputK, delimiter = "\t", fmt = "%s")

    
