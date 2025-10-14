import numpy as np
import matplotlib.pyplot as plt
import pyablo
from scipy.optimize import newton
import sys

from configparser import ConfigParser

## Solution to the SOD problem 
class sod_analytical:
  def __init__(self, rL, uL, pL, rR, uR, pR, xi, T, gg ):
    # rL, uL, pL, rR, uR, pR : Initial conditions of the Reimann problem 
    # xi: POSITION where the interface sits.
    # T: the desired solution time
    # gg: adiabatic constant 1.4=7/5 for a 3D diatomic gas    
    self.rL = rL
    self.uL = uL
    self.pL = pL
    self.uR = uR
    self.pR = pR
    self.xi = xi
    self.T = T
    # compute speed of sound
    cL = np.sqrt(gg*pL/rL); 
    cR = np.sqrt(gg*pR/rR);
    # compute P
    # Function to find the roots of!
    def f(P, pL, pR, cL, cR, gg):
      a = (gg-1)*(cR/cL)*(P-1) 
      b = np.sqrt( 2*gg*(2*gg + (gg+1)*(P-1) ) )
      return P - pL/pR*( 1 - a/b )**(2.*gg/(gg-1.))
    self.P = newton(f, 0.5, args=(pL, pR, cL, cR, gg), tol=1e-12);

  def value( self, x ):  
    rL = self.rL
    uL = self.uL
    pL = self.pL
    uR = self.uR
    pR = self.pR
    xi = self.xi
    T  = self.T     
    P  = self.P     
    
    # compute speed of sound
    cL = np.sqrt(gg*pL/rL); 
    cR = np.sqrt(gg*pR/rR);

    # compute region positions right to left
    # region R
    c_shock = uR + cR*np.sqrt( (gg-1+P*(gg+1)) / (2*gg) )
    x_shock = xi + T * c_shock
    if( x >= x_shock ):
        return (rR, uR, pR, 5)

    # region 2
    alpha = (gg+1)/(gg-1)
    c_contact = uL + 2*cL/(gg-1)*( 1-(P*pR/pL)**((gg-1.)/2/gg) )
    x_contact = xi + T * c_contact

    if( x >= x_contact ):
        r = (1 + alpha*P)/(alpha+P)*rR
        u = c_contact
        p = P * pR
        return ( r, u, p, 4 )

    # region 3
    r3 = rL*(P*pR/pL)**(1/gg);
    p3 = P*pR;
    c_fanright = c_contact - np.sqrt(gg*p3/r3)
    x_fanright = xi + c_fanright*T

    if( x >= x_fanright ):
        r = r3
        u = c_contact
        p = P * pR
        return ( r, u, p, 3 )

    # region 4
    c_fanleft = -cL
    x_fanleft = xi + c_fanleft*T
    if( x >= x_fanleft ):
        u = 2 / (gg+1) * (cL + (x-xi)/T )
        r = rL*(1 - (gg-1)/2.*u/cL)**(2/(gg-1))
        p = pL*(1 - (gg-1)/2.*u/cL)**(2*gg/(gg-1))
        return ( r, u, p, 2 )

    return (rL,uL,pL,1)

config = ConfigParser(inline_comment_prefixes=('#',';'))
config.read('last.ini')

# Physics properties
rL = config.getfloat('sod', 'rhoL')
pL = config.getfloat('sod', 'pL')
uL = 0.0

rR = config.getfloat('sod', 'rhoR')
pR = config.getfloat('sod', 'pR')
uR = 0.0

gg = config.getfloat('hydro', 'gamma0') 
#xi = config.getfloat('sod', 'x0')
T  = config.getfloat('run', 'tend')

## Mesh properties
xmin  = config.getfloat('mesh', 'xmin')
xmax  = config.getfloat('mesh', 'xmax')
ymin  = config.getfloat('mesh', 'ymin')
ymax  = config.getfloat('mesh', 'ymax')
angle = config.getfloat('sod', 'angle')

## Amr and number of cells max
bx = config.getint('amr', 'bx')
lvlmax = config.getint('amr', 'level_max')
Nx = 2**lvlmax * bx
dx = (xmax-xmin)/Nx

## The ray over which we project the data
ddx = -np.sin(angle-np.pi*0.5)
ddy =  np.cos(angle-np.pi*0.5)

## Casting the ray from the top right corner
x = np.linspace(xmin+0.5*dx, xmax-0.5*dx, Nx)
xd = x-0.5*(xmin+xmax)
y = 0.5*(xmin+xmax) + xd * ddy/ddx
## Reading dyablo run
pos = np.stack((x, y, np.zeros_like(x))).T

reader = pyablo.XdmfReader()
snap_list = reader.readTimeSeries('rotated_sod_main.xmf')
snap_id = snap_list[-1]
#snap_id = snap_list[0]
snap = reader.readSnapshot(snap_id)
T = snap.getTime()

rho    = np.array(snap.probeDensity(pos))
rho_vv = np.array(snap.probeMomentum(pos))
e_tot  = np.array(snap.probeEnergy(pos))

u = rho_vv[:,0]/rho
v = rho_vv[:,1]/rho

vv2 = u**2.0 + v**2.0
vv = np.sqrt(vv2)

Ek = 0.5 * rho * vv2
p  = (e_tot - Ek) * (gg-1.0)
xx = np.sqrt(x**2.0 + y**2.0)

## Center of the position array should be mapped to 0.5
xx -= xx[Nx//2]

## Extracting expected values

#formula = sod_analytical(rL, uL, pL, rR, uR, pR, 0.5, T, gg)
formula = sod_analytical(rL, uL, pL, rR, uR, pR, 0.0, T, gg)

analytical = np.asarray([ formula.value( x ) for x in xx ])

expected_rho = analytical[:,0]
expected_u = analytical[:,1]
expected_prs = analytical[:,2]

## Computing errors
L1_rho = np.linalg.norm(rho - expected_rho, ord=1) / Nx
L1_u   = np.linalg.norm(vv - expected_u, ord=1)   / Nx
L1_prs = np.linalg.norm(p - expected_prs, ord=1) / Nx

print( f'Density; L1 error = {L1_rho:.4}' )
print( f'Velocity; L1 error = {L1_u:.4}' )
print( f'Pressure; L1 error = {L1_prs:.4}' )

## And plotting
png_filename = 'rotated_sod.png'
fig, ax = plt.subplots(3, 1, figsize=(7, 7))
ax[0].plot(xx, expected_rho, '--k')
ax[0].plot(xx, rho, '-r')
ax[0].set_title(f'Density; L1 error = {L1_rho:.4}')
ax[0].set_xlabel('x')
ax[0].set_ylabel(r'$\rho$')
ax[1].plot(xx, expected_u, '--k')
ax[1].plot(xx, vv, '-r')
ax[1].set_title(f'Velocity; L1 error = {L1_u:.4}')
ax[1].set_xlabel('x')
ax[1].set_ylabel(r'$v_x$')
ax[2].plot(xx, expected_prs, '--k')
ax[2].plot(xx, p, '-r')
ax[2].set_title(f'Pressure; L1 error = {L1_prs:.4}')
ax[0].set_xlabel('x')
ax[0].set_ylabel(r'$p$')
plt.suptitle(f'SOD Shock tube')
plt.tight_layout()
plt.savefig(f'{png_filename}')
print( f'Exported {png_filename}' )
plt.show()
