# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:46:34 2022

projected gradient method for opimizing system controllbability
by searching hypershperes of constant radius

@author: curha
"""

import numpy as np
from scipy.linalg import expm, inv, pinv
from progress.bar import Bar

class PGM():
    def __init__(self,eta=0.001,epsilon=0.01,radius=10,convergence=1e-3,maxiter=1000,norm = 1):
        
        self.eta = eta
        self.epsilon = epsilon
        self.radius = radius
        self.convergence = convergence
        self.maxiter = maxiter
        self.norm = norm
        
        return
        
    def initialize(self,A,num_in):
        
        # initialize input gain from right eigenvecs of A (maximizes 
        # controllability subspace dimension)
        self.A=A
        self.num_in = num_in
        self.zero = np.zeros(A.shape)
        self.expA = expm(np.transpose(A))
        vals,vecs = np.linalg.eig(A)
        self.B_init = vecs[:,:num_in]
        Binner = np.trace(np.inner(self.B_init,self.B_init))
        Brt = np.sqrt(Binner)
        self.B_init = np.sqrt(self.radius+self.epsilon)*self.B_init/Brt
    
    def Grammian(self, B):
        # compute grammian
        Bouter = B@np.transpose(B)
        M = np.block([[-self.A, Bouter],[self.zero, np.transpose(self.A)]])
        expM = expm(M)
        G = np.transpose(self.expA)@expM[:self.A.shape[0],self.A.shape[0]:]
        
        return G
    
    def gradients(self, B, G):
        
        # compute gradients
        Ginv = inv(G)

        dNdB = 4*(np.trace(np.transpose(B)@B)-self.radius)*B
        #determinant(i) = det(G)
        M = np.block([[-np.transpose(self.A), Ginv],[self.zero, self.A]])
        expM = expm(M)
        L = np.transpose(self.expA)@expM[:self.A.shape[0],self.A.shape[0]:]
        dEdB = -L@B
        
        return np.resize(dNdB,np.prod(dNdB.shape)), np.resize(dEdB,np.prod(dEdB.shape))
    
    def update(self, B):
        
        
        G = self.Grammian(B)
        
        dNdB, dEdB = self.gradients(B,G)
        if self.norm:
            dNdB = dNdB/np.linalg.norm(dNdB)
            dEdB = dEdB/np.linalg.norm(dEdB)
            
        # get projection operator
        proj = np.eye(dNdB.shape[0]) - dNdB@pinv(dNdB[np.newaxis,:])

        proj = self.eta*proj@dEdB
        
        # update input gain matrix
        tmp = B - np.reshape(proj,B.shape)
        B = np.sqrt(self.radius+self.epsilon)*tmp/np.linalg.norm(tmp)
        c = 1 + np.inner(dNdB,dEdB)/(np.linalg.norm(dNdB)*np.linalg.norm(dEdB))
        
        return B, G, c
        
    def optimize(self):
        
        B = self.B_init
        C = np.asarray(1)
        c = C
        i = 0
        with Bar('Processing...') as bar:
            while c > self.convergence:
            
                #for i in range(self.maxiter):
            
                B, G, c = self.update(B)

                i += 1
                C=np.append(C,c)
                # provide feedback for the user
                status = 1-(c - self.convergence)/(C[0]-self.convergence)
                #breakpoint()
                bar.goto(np.round(status*100))
                
                if i > self.maxiter:
                    break 
            
        return B, G, C
    
class ERA():
    # eigensystem realization algorithm
    def __init__(self):
        

        
        return


class sysID():
    # observer Kalman filter identification
    def __init__(self,y,u, r, lam=1e-5):
        self.y=y
        self.u=u
        self.lam=lam
        # Step 0, check shapes of y,u
        yshape = y.shape
        self.q = yshape[0]    # q is the number of outputs
        self.l = yshape[1]    # L is the number of output samples
        ushape = u.shape
        self.m = ushape[0]    # m is the number of inputs
        self.lu = ushape[1]   # Lu i the number of input samples
        assert(self.l==self.lu)    # L and Lu need to be the same length

        # Step 1, p is the number of estimated markov params 
        self.p = r

        # Step 2, form data matrices y and V as shown in Eq. (7),
        # solve for observer Markov parameters, Ybar
        self.V = np.zeros([self.m + (self.m+self.q)*self.p,self.l])
        
        return
    
    def OKID(self):
        
        V=self.V
        for i in range(self.l):
            V[:self.m,i] = self.u[:self.m,i]
            
        for i in range(1,self.p):
            for j in range(self.l-i):
                vtemp = np.block([self.u[:,j],self.y[:,j]])
                V[self.m+(i-1)*(self.m+self.q):self.m+i*(self.m+self.q),i+j] = vtemp

        Ybar = self.y@pinv(V, self.lam)
        
        # Step 3, isolate system Markov parameters H, and observer gain M
        D = Ybar[:,:self.m]  # feed-through term (or D matrix) is the first term
        YbarNoD = Ybar[:,self.m:]
        
        Y = np.zeros([self.q,self.m,YbarNoD.shape[0]])
        Ybar1 = np.zeros([self.q,self.m,YbarNoD.shape[0]])
        Ybar2 = np.zeros([self.q,self.q,YbarNoD.shape[0]])
        for i in range(self.p):
            Ybar1[:,:,i] = YbarNoD[:,(self.m+self.q)*i:(self.m+self.q)*i+self.m]
            Ybar2[:,:,i] = YbarNoD[:,(self.m+self.q)*i+self.m:(self.m+self.q)*i+self.m+self.q]
        
        Y[:,:,0] = Ybar1[:,:,0] + Ybar2[:,:,0]@D
        for k in range(1,self.p):
            Y[:,:,k] = Ybar1[:,:,k] + Ybar2[:,:,k]@D
            for i in range(k):
                Y[:,:,k] = Y[:,:,k] + Ybar2[:,:,i]@Y[:,:,k-i]
        
        H = D
        for k in range(self.p):
            H = np.concatenate((H, Y[:,:,k]),axis=1)

        self.H = H
        return H
    
    def ERA(self, H, cut):
        # eigensystem realization algorithm
        H0d = H[:,self.m:self.m*(self.p-2)]
        H1d = H[:,self.m*2:self.m*(self.p-1)]
        
        R, D, S = np.linalg.svd(H0d)
        VarExplained = np.cumsum(D)/np.sum(D)
        Dsv = np.diag(np.sqrt(D[:cut]))
        S=np.transpose(S)
        Rn = R[:,:cut]
        Sn = S[:,:cut]
        Dinv = np.diag(np.sqrt(1/D[:cut]))
        A = Dinv@np.transpose(Rn)@H1d@Sn@Dinv
        B = Dsv@np.transpose(Sn)
        B = B[:,:self.m]
        C = Rn@Dsv
        C = C[:self.q,:]
        D = H[:,:self.m]
        
        return A,B,C,D
    
    def map(self,Ae,Be,Ce,C):
        # linear mapping from reduced order model to full LTI state-space model
        psiinv = pinv(Ce)@C
        psi = pinv(psiinv)
        Aest = psi@Ae@psiinv
        Best = psi@Be
        Cest = Ce@psiinv
        
        return psi,psiinv,Aest,Best,Cest
        
        