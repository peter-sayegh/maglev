from sympy import symbols, diff, solve, sqrt, pi
import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy, scipy.optimize
from scipy import special
from sympy import symbols, diff, lambdify, Function
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_k

# Constants
o = 100 #number of guideway coil
k = 50
I = 7* 10**7
L = 1
v = 160
dz_0 = 0
Mass=25*10**3/(24)

R=0.5
r=0.25
L=0.6
mu_0=1.26*(10**(-6))
R_E=0.1

M=I*np.pi*R**2

def xk(t, i):
    return 2*R*i+v*t

def d(t,i,xc,zc,z0):
    return np.sqrt( (xk(t,i)-xc)**2 + (z0-zc)**2)

def d_dot(t,i,xc,zc,z0):
    #print("xk:",xk(t,i))

    return v*(xk(t,i)-xc)/d(t,i,xc,zc,z0)


def p(t,i,xc,zc,z0):
    return np.sqrt(4*r*d(t,i,xc,zc,z0)/(L**2+(d(t,i,xc,zc,z0)+r)**2))

def p_dot(t,i,xc,zc,z0):
    return 2*r*d_dot(t,i,xc,zc,z0) * (1 - 2*d(t,i,xc,zc,z0)*(r+d(t,i,xc,zc,z0))/(L**2+(d(t,i,xc,zc,z0)+r)**2)) / ( p(t,i,xc,zc,z0)*(L**2+(r+d(t,i,xc,zc,z0))**2))

def E(p):
    """Complete elliptic integral of the second kind."""
    return special.ellipe(p)

def K(p):
    """Complete elliptic integral of the first kind."""
    return special.ellipk(p)

def E_dot(p, p_dot):
    return p_dot/p * (E(p)-K(p))
def K_dot(p, p_dot):
    return p_dot/p * (E(p)/(1-p**2) - K(p))
    


A=mu_0*M/(8*np.pi)


def B(t,i,xc,zc,z0):
    return p(t,i,xc,zc,z0)/np.sqrt(r*d(t,i,xc,zc,z0)**3)
    
def B_dot(t,i,xc,zc,z0):
    return ( p_dot(t,i,xc,zc,z0)*np.sqrt(r*d(t,i,xc,zc,z0)**3) - (3*p(t,i,xc,zc,z0)*r*d_dot(t,i,xc,zc,z0)*d(t,i,xc,zc,z0)**2)/(2*np.sqrt(r*d(t,i,xc,zc,z0)**3)))/(r*d(t,i,xc,zc,z0)**3)
            

def C(t,i,xc,zc,z0):
    return -(2*d(t,i,xc,zc,z0)+p(t,i,xc,zc,z0)**2*(r-d(t,i,xc,zc,z0)))*E(p(t,i,xc,zc,z0))/(1-p(t,i,xc,zc,z0)**2)

def C_dot(t,i,xc,zc,z0):
    return -(2*d_dot(t,i,xc,zc,z0)+2*p_dot(t,i,xc,zc,z0)*p(t,i,xc,zc,z0)*(r-d(t,i,xc,zc,z0))-d_dot(t,i,xc,zc,z0)*p(t,i,xc,zc,z0)**2)*E(p(t,i,xc,zc,z0))/(1-p(t,i,xc,zc,z0)**2) - (E_dot(p(t,i,xc,zc,z0),p_dot(t,i,xc,zc,z0))*(1-p(t,i,xc,zc,z0)**2)+2*p(t,i,xc,zc,z0)*p_dot(t,i,xc,zc,z0)*E(p(t,i,xc,zc,z0)))*(2*d(t,i,xc,zc,z0)+p(t,i,xc,zc,z0)**2*(r-d(t,i,xc,zc,z0)))/(1-p(t,i,xc,zc,z0)**2)**2

            
def D(t,i,xc,zc,z0):
    return 2*d(t,i,xc,zc,z0)*K(p(t,i,xc,zc,z0))/p(t,i,xc,zc,z0)**2

def D_dot(t,i,xc,zc,z0):
    return ((2*d_dot(t,i,xc,zc,z0)*K(p(t,i,xc,zc,z0)) + 2*d(t,i,xc,zc,z0)*K_dot(p(t,i,xc,zc,z0),p_dot(t,i,xc,zc,z0)))*p(t,i,xc,zc,z0)**2 - 2*d(t,i,xc,zc,z0)*K(p(t,i,xc,zc,z0))*2*p(t,i,xc,zc,z0)*p_dot(t,i,xc,zc,z0))/p(t,i,xc,zc,z0)**4




def dphi(t,i,xc,zc,z0):
    return A *( B_dot(t,i,xc,zc,z0) * (C(t,i,xc,zc,z0)+D(t,i,xc,zc,z0)) + B(t,i,xc,zc,z0)*( C_dot(t,i,xc,zc,z0) + D_dot(t,i,xc,zc,z0) ))

def Current(t,xc,z0):
    c=0
    for i in range(-k,k):
        c+= (-1)**i * (dphi(t,i,xc,0,z0)-dphi(t,i,xc,2*r,z0))/R_E
    
    return c





z0=r-0.01

T=np.linspace(0,0.04,100)
a=np.array([Current(t,0,z0) for t in T])

plt.figure(figsize=(8, 5))
plt.plot(T,a, color='black', label='Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Current as a Function of Time')
plt.legend()
plt.grid(True)
plt.show()


#Current
#_____________________________________________________________________________________
#Magnetic Field





def Ugrav(z_0):
    return Mass*9.81*(z_0)


def total_field(t,xC,z_0): #specifically for R=4*r
    temp1 = 0
    temp2 = 0
    current=[]
    for i in range(4):
        current.append(Current(t,2*r*i,z_0))
    
    
    for j in range(-o,o+1):
        #print(j, xC, z_0)
        s=j%4
        #print(s)
        relative_pos=2*r*j-xC
        i = current[s]
        Mguideway = i*math.pi*r**2
        
        norm1=np.sqrt((v*t-relative_pos)**2 + (L)**2 + (z_0-2*r)**2)
        norm2=np.sqrt((v*t-relative_pos)**2 + (L)**2 + (z_0)**2)
        #if j==-o:
            #print(norm1, norm2)
        
        coef1 = (3 * Mguideway *L) / ( norm1 ** 5)
        coef2 = (3 * Mguideway *L) / ( norm2 ** 5)
        temp1 += (coef1 * L - (1/(norm1)**3) * Mguideway)
        temp2 += (coef2 * L - (1/(norm2)**3) * Mguideway)

    return mu_0/(4*math.pi)*(temp1+temp2)




# Calculate potential energy
z_values = np.linspace(-r, 3*r, 100)
t=0.4*R/v
b_y = np.array([(total_field(t,0,z)/(2*o+1)-total_field(t,2*R,z)/(2*o+1))/2 for z in z_values]) 














Ug = Ugrav(z_values)
U = []

for i in range(len(b_y)):
    U.append(-M*b_y[i]+Ug[i])
    
U = np.array(U)

min_index = np.argmin(U)
min_height = z_values[min_index]
min_energy = U[min_index]
print(min_height)

plt.figure(figsize=(8, 5))
plt.plot(z_values, -M*b_y, label='U_mag', color='blue', linewidth=1.5)
plt.plot(z_values, U, label='U_tot', color='black', linewidth=1.5)
plt.plot(z_values, Ug, label='U_grav', color='red', linestyle='--', linewidth=1.5)
plt.plot(min_height, min_energy, 'ro', markersize=10)

plt.xlabel('Height of the Train (m)', fontsize=23)
plt.ylabel('Energy (J)', fontsize=23)
plt.legend(fontsize=18)
plt.grid(True)
plt.show()

    
    