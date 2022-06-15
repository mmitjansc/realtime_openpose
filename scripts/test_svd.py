#!/usr/bin/env python

import numpy as np

def main():


    # Create plane coefficients
    A = 3
    B = -2
    C = 1
    D = 4


    X = np.random.randn(7,3) # 7 joints, 2 channels
    # We need to find the 3rd channel by: Ax+By+Cz+d=0
    z = (-D-A*X[:,0]-B*X[:,1])/C
    # Add some noise to Z:
    z += (np.random.rand(*z.shape) - 0.5) * 1
    X[:,2] = z

    

    # Let's find the perpendicular vector to the plane, and make sure it's perpendicular to everything:
    v = np.array([A,B,C],dtype=float); 
    
    # Prepare to run SVD on the covariance matrix:
    mean = X.mean(axis=0,keepdims=True)
    Cov = (X-mean).T.dot(X-mean) / (X.shape[0]-1)

    # Run singular value decomposition:
    U,S,Vt = np.linalg.svd(Cov)
    V = Vt.T
    print("Theoretical perpendicular vector:",v/np.linalg.norm(v))
    print("Practical perpendicular vector:",Vt[2,:])

    # Find the plane coefficients:
    A,B,C = V[:,2]
    D = -(A*X[:,0] + B*X[:,1] + C*X[:,2]).mean()
    plane = np.append(V[:,2],D)

    print("Obtained plane:",plane)

    ### Now, with new X/Z, Y/Z, find their XYZ coordinates
    # Camera calibration matrix:
    K = np.array([421.7107238769531, 0.0, 426.1136169433594, 0.0, 420.7950744628906, 239.36465454101562, 0.0, 0.0, 1.0]).reshape((3,3))
    xp_yp = np.random.randn(10,2) * 10
    unit_xyz = np.linalg.inv(K).dot(np.concatenate((xp_yp,np.ones((xp_yp.shape[0],1))),axis=1).T).T

    # Finally, obtain the XYZ coordinates:
    z = -D/(A*unit_xyz[:,[0]] + B*unit_xyz[:,[1]] + C)
    x = unit_xyz[:,[0]]*z
    y = unit_xyz[:,[1]]*z
    point_coord = np.concatenate((x,y,z),axis=1)
    print("XYZ coordinates:\n",point_coord)

    ## Test if they actually belong to the plane...
    print(np.concatenate((point_coord,np.ones((point_coord.shape[0],1))),axis=1).dot(plane))



main()