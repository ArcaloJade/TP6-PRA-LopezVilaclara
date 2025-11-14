# -*- coding: utf-8 -*-
"""
ICP framework

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time


def closest_point_matching(X, P):
  """Performs closest point matching of two point sets.
  
  Arguments:
  X -- reference point set
  P -- point set to be matched with the reference
  
  Output:
  P_matched -- reordered P, so that the elements in P match the elements in X
  """
  
  P_matched = P

  #TODO: implement
    
  return P_matched
  
  
def plot_icp(X, P, P0, i, e):
  
  plt.cla()
  plt.scatter(X[0,:],X[1,:],c='b',marker='o',s=50)
  plt.scatter(P[0,:],P[1,:],c='m',marker='o',s=50)
  plt.scatter(P0[0,:],P0[1,:],c='r',marker='o',s=50)
  plt.legend(('X','P','P0'),loc='lower left')
  plt.plot(np.vstack((X[0,:],P[0,:])),np.vstack((X[1,:],P[1,:])),c='k')
  plt.title("iteration = "+str(i)+"   rmse = "+str(e))
  plt.axis([-10,15,-10,15])
  plt.gca().set_aspect('equal',adjustable='box')
  plt.draw()
  time.sleep(0.5)
  
  
def icp(X, P, matching_flag=True):
  
  P0 = P
  
  plt.figure()
  plt.ion()
  plt.show()
  
  for i in range(0,15):
    
    #calculate RMSE
    e = 0
    for j in range(0,P.shape[1]):
      e = e+math.pow(P[0,j]-X[0,j],2)+math.pow(P[1,j]-X[1,j],2)
    e = math.sqrt(e/P.shape[1])
    
    #plot icp
    plot_icp(X,P,P0,i,e)
    
    #data association
    if matching_flag:
      P = closest_point_matching(X, P)
    
    #substract center of mass
    mx = np.transpose([np.mean(X,1)])
    mp = np.transpose([np.mean(P,1)])
    X_prime = X-mx
    P_prime = P-mp
    
    #singular value decomposition
    W = np.dot(X_prime,np.transpose(P_prime))
    U, s, V = np.linalg.svd(W)
    
    #calculate rotation and translation
    R = np.dot(U,V)
    
    #TODO: Check for reflection and correct if necessary
    
    t = mx-np.dot(R,mp)
    
    #apply transformation
    P = np.dot(R,P)+t
    
  plt.ioff()
  plt.show()
    
    
def main():
  
  #create reference data  
  X = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,-1,-2,-3,-4,-5]])
    
  #add noise
  P = X+0.05*np.random.normal(0,1,X.shape)
  
  #translate
  P[0,:] = P[0,:]+1
  P[1,:] = P[1,:]+1
  
  #rotate
  theta1 =  10.0/360*2*np.pi
  theta2 = 110.0/360*2*np.pi
  rot1 = np.array([[math.cos(theta1),-math.sin(theta1)],
                   [math.sin(theta1),math.cos(theta1)]])
  rot2 = np.array([[math.cos(theta2),-math.sin(theta2)],
                   [math.sin(theta2),math.cos(theta2)]])
  
  #sets with known correspondences
  P1 = np.dot(rot1,P)
  P2 = np.dot(rot2,P)
  
  #sets with unknown correspondences
  P3 = np.transpose(np.random.permutation(np.transpose(P1)))
  P4 = np.transpose(np.random.permutation(np.transpose(P2)))
  
  #execute icp
  icp(X,P1,False)
  #icp(X,P2,False)
  #icp(X,P3,True)
  #icp(X,P4,True)
    
if __name__ == "__main__":
  main()
