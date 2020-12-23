import math as m
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt

# INPUTS:
ds = 1.0
numRings = 1000
diam = 10.0
steps = round(numRings * diam * 6.2832 / ds) + 1
traceX = np.zeros(steps);
traceY = np.zeros(steps);
traceX[0] = diam
traceY[0] = 0

# returns vectors following contours of a circle
def getVector(x, y):
    norm = m.sqrt(x**2 + y**2)
    if norm == 0:
        vector = [0, 0]
    else:
        # CCW
        #vector = [-y / norm, x / norm]
        # CW
        vector = [y / norm, -x / norm]
    return vector

# interpolates the vector field between discrete points
def interpField(vectorField, x, y):
    x0 = m.floor(x)
    x1 = m.ceil(x)
    y0 = m.floor(y)
    y1 = m.ceil(y)
    
    dx = x - x0
    dy = y - y0
    
    weight00 = (1 - dx) * (1 - dy)
    weight01 = (1 - dx) * dy
    weight10 = dx * (1 - dy)
    weight11 = dx * dy
    
    fieldX = weight00 * vectorField[x0][y0][0] + weight01 * vectorField[x0][y1][0] + weight10 * vectorField[x1][y0][0] + weight11 * vectorField[x1][y1][0]
    fieldY = weight00 * vectorField[x0][y0][1] + weight01 * vectorField[x0][y1][1] + weight10 * vectorField[x1][y0][1] + weight11 * vectorField[x1][y1][1]
    
    return [fieldX, fieldY]

# create the vector field
vectorField = np.zeros((41, 41, 2))
vectorX = np.zeros(1681)
vectorY = np.zeros(1681)
vectorU = np.zeros(1681)
vectorV = np.zeros(1681)
for x in range(-20, 21):
    for y in range(-20, 21):
        vector = getVector(x, y)
        vectorField[x][y] = vector
        index = (x + 20) * 41 + y
        vectorX[index] = x;
        vectorY[index] = y;
        vectorU[index] = vector[0]
        vectorV[index] = vector[1]

# plot the vector field
plt.figure(figsize = (9, 9))
plt.quiver(vectorX, vectorY, vectorU, vectorV, scale = 50)
plt.axes().set_aspect('equal')  
plt.tight_layout()

#----- First Order -----------------------------------------------------------#
#for i in range (1, steps):
#    xn = traceX[i - 1]
#    yn = traceY[i - 1]
#    
#    field = interpField(vectorField, xn, yn)
#    
#    traceX[i] = xn + field[0] * ds
#    traceY[i] = yn + field[1] * ds
#
#print(traceX[len(traceX) - 1])
#-----------------------------------------------------------------------------#


#----- Second Order ----------------------------------------------------------#
## calculate second point first order
#x0 = traceX[0]
#y0 = traceY[0]
#field = interpField(vectorField, x0, y0)
#x1 = x0 + field[0] * ds
#y1 = y0 + field[1] * ds
#traceX[1] = x1
#traceY[1] = y1
#    
#for i in range (2, steps):
#    xn_1 = traceX[i - 2]
#    yn_1 = traceY[i - 2]
#    xn = traceX[i - 1]
#    yn = traceY[i - 1]
#    
#    field_1 = interpField(vectorField, xn_1, yn_1)
#    field = interpField(vectorField, xn, yn)
#    dFieldX = (field[0] - field_1[0]) / ds
#    dFieldY = (field[1] - field_1[1]) / ds
#    
#    traceX[i] = xn + field[0] * ds + (1 / 2) * dFieldX * ds**2
#    traceY[i] = yn + field[1] * ds + (1 / 2) * dFieldY * ds**2
#
#print(traceX[len(traceX) - 1])
#-----------------------------------------------------------------------------#
 
   
#----- Third Order -----------------------------------------------------------#
## calculate second point first order, third point second order
#x0 = traceX[0]
#y0 = traceY[0]
#field_1 = interpField(vectorField, x0, y0)
#x1 = x0 + field_1[0] * ds
#y1 = y0 + field_1[1] * ds
#traceX[1] = x1
#traceY[1] = y1
#
#field = interpField(vectorField, x1, y1)
#dFieldX = (field[0] - field_1[0]) / ds
#dFieldY = (field[1] - field_1[1]) / ds
#x2 = x1 + field[0] * ds + (1 / 2) * dFieldX * ds**2
#y2 = y1 + field[1] * ds + (1 / 2) * dFieldY * ds**2
#traceX[2] = x2
#traceY[2] = y2
#
#for i in range (2, steps):
#    xn_2 = traceX[i - 3]
#    yn_2 = traceY[i - 3]
#    xn_1 = traceX[i - 2]
#    yn_1 = traceY[i - 2]
#    xn = traceX[i - 1]
#    yn = traceY[i - 1]
#    
#    field_2 = interpField(vectorField, xn_2, yn_2)
#    field_1 = interpField(vectorField, xn_1, yn_1)
#    field = interpField(vectorField, xn, yn)
#    dFieldX_1 = (field_1[0] - field_2[0]) / ds
#    dFieldY_1 = (field_1[1] - field_2[1]) / ds
#    dFieldX = (field[0] - field_1[0]) / ds
#    dFieldY = (field[1] - field_1[1]) / ds
#    ddFieldX = (dFieldX - dFieldX_1) / ds
#    ddFieldY = (dFieldY - dFieldY_1) / ds
#    
#    traceX[i] = xn + field[0] * ds + (1 / 2) * dFieldX * ds**2 + (1 / 6) * ddFieldX * ds**3
#    traceY[i] = yn + field[1] * ds + (1 / 2) * dFieldY * ds**2 + (1 / 6) * ddFieldY * ds**3
#    
#print(traceX[len(traceX) - 1])
#-----------------------------------------------------------------------------#

#----- RK4 -------------------------------------------------------------------#
# RK4 is self starting no need to calculate initial points
for i in range (1, steps):
    X = traceX[i - 1]
    Y = traceY[i - 1]
    
    Slopes = interpField(vectorField, X, Y);
    K1x = Slopes[0];
    K1y = Slopes[1];
    
    Slopes2 = interpField(vectorField, X + (ds/2)*K1x, Y + (ds/2)*K1y);
    K2x = Slopes2[0];
    K2y = Slopes2[1];
    
    Slopes3 = interpField(vectorField, X + (ds/2)*K2x, Y + (ds/2)*K2y);
    K3x = Slopes3[0];
    K3y = Slopes3[1];
    
    Slopes4 = interpField(vectorField, X + ds*K3x, Y + ds*K3y);
    K4x = Slopes4[0];
    K4y = Slopes4[1];
    
    traceX[i] = X + (ds/6)*(K1x + 2*K2x + 2*K3x + K4x);
    traceY[i] = Y + (ds/6)*(K1y + 2*K2y + 2*K3y + K4y);
    
print(traceX[len(traceX) - 1])
#-----------------------------------------------------------------------------#

# plot the trace    
plt.plot(traceX, traceY)
plt.axes().set_aspect('equal')


        