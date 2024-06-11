import jax
jax.config.update("jax_enable_x64", True)   # turn this on to use double precision JAX
import jax.numpy as jnp
from jax import jit, grad
from jaxopt import ScipyMinimize

#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


"""
Create Your Own Finite Volume Fluid Simulation (With JAX)
Philip Mocz (2024) @PMocz

Simulate the compressible isothermal Euler equations in 2D using a finite volume method.
Use autodiff and plug the simulation into an optimization loop to solve for the 
initial conditions that lead to a desired final state.

"""

R = -1   # right
L = 1    # left
aX = 1   # x-axis
aY = 0   # y-axis


@jit
def getConserved( rho, vx, vy, vol ):
  """
    Calculate the conserved variable from the primitive
  rho      is matrix of cell densities
  vx       is matrix of cell x-velocity
  vy       is matrix of cell y-velocity
  vol      is cell volume
  Mass     is matrix of mass in cells
  Momx     is matrix of x-momentum in cells
  Momy     is matrix of y-momentum in cells
  """
  Mass   = rho * vol
  Momx   = rho * vx * vol
  Momy   = rho * vy * vol
  
  return Mass, Momx, Momy


@jit
def getPrimitive( Mass, Momx, Momy, vol ):
  """
    Calculate the primitive variable from the conservative
  Mass     is matrix of mass in cells
  Momx     is matrix of x-momentum in cells
  Momy     is matrix of y-momentum in cells
  vol      is cell volume
  rho      is matrix of cell densities
  vx       is matrix of cell x-velocity
  vy       is matrix of cell y-velocity
  """
  rho = Mass / vol
  vx  = Momx / rho / vol
  vy  = Momy / rho / vol

  return rho, vx, vy


@jit
def getGradient(f, dx):
  """
    Calculate the gradients of a field
  f        is a matrix of the field
  dx       is the cell size
  f_dx     is a matrix of derivative of f in the x-direction
  f_dy     is a matrix of derivative of f in the y-direction
  """

  f_dx = ( jnp.roll(f,R,axis=aX) - jnp.roll(f,L,axis=aX) ) / (2*dx)
  f_dy = ( jnp.roll(f,R,axis=aY) - jnp.roll(f,L,axis=aY) ) / (2*dx)
  
  return f_dx, f_dy


@jit
def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
  """
    Calculate the gradients of a field
  f        is a matrix of the field
  f_dx     is a matrix of the field x-derivatives
  f_dy     is a matrix of the field y-derivatives
  dx       is the cell size
  f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis 
  f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis 
  f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis 
  f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis 
  """

  f_XL = f - f_dx * dx/2
  f_XL = jnp.roll(f_XL,R,axis=aX)
  f_XR = f + f_dx * dx/2
  
  f_YL = f - f_dy * dx/2
  f_YL = jnp.roll(f_YL,R,axis=aY)
  f_YR = f + f_dy * dx/2
  
  return f_XL, f_XR, f_YL, f_YR
  

@jit
def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
  """
    Apply fluxes to conserved variables
  F        is a matrix of the conserved variable field
  flux_F_X is a matrix of the x-dir fluxes
  flux_F_Y is a matrix of the y-dir fluxes
  dx       is the cell size
  dt       is the timestep
  """
  
  # update solution
  F += - dt * dx * flux_F_X
  F +=   dt * dx * jnp.roll(flux_F_X,L,axis=aX)
  F += - dt * dx * flux_F_Y
  F +=   dt * dx * jnp.roll(flux_F_Y,L,axis=aY)
  
  return F


@jit
def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R):
  """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
  rho_L        is a matrix of left-state  density
  rho_R        is a matrix of right-state density
  vx_L         is a matrix of left-state  x-velocity
  vx_R         is a matrix of right-state x-velocity
  vy_L         is a matrix of left-state  y-velocity
  vy_R         is a matrix of right-state y-velocity
  flux_Mass    is the matrix of mass fluxes
  flux_Momx    is the matrix of x-momentum fluxes
  flux_Momy    is the matrix of y-momentum fluxes
  """

  # compute star (averaged) states
  rho_star  = 0.5*(rho_L + rho_R)
  momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
  momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
  
  P_star = rho_star
  
  # compute fluxes (local Lax-Friedrichs/Rusanov)
  flux_Mass   = momx_star
  flux_Momx   = momx_star**2/rho_star + P_star
  flux_Momy   = momx_star * momy_star/rho_star
  
  # find wavespeeds
  C_L = 1 + jnp.abs(vx_L)
  C_R = 1 + jnp.abs(vx_R)
  C = jnp.maximum( C_L, C_R )
  
  # add stabilizing diffusive term
  flux_Mass   -= C * 0.5 * (rho_L - rho_R)
  flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
  flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)

  return flux_Mass, flux_Momx, flux_Momy


@jit
def update_sim(i, values):
  """
    Take a simulation step
  """
  rho, vx, vy, Mass, Momx, Momy, dx, dt  = values

  vol = dx**2
  
  # get Primitive variables
  rho, vx, vy = getPrimitive( Mass, Momx, Momy, vol )
  
  # get time step (CFL) = dx / max signal speed
  #dt = courant_fac * jnp.min( dx / (1.0 + jnp.sqrt(vx**2+vy**2)) )
  
  # calculate gradients
  rho_dx, rho_dy = getGradient(rho, dx)
  vx_dx,  vx_dy  = getGradient(vx,  dx)
  vy_dx,  vy_dy  = getGradient(vy,  dx)
  P_dx = rho_dx
  P_dy = rho_dy
  
  # extrapolate half-step in time
  rho_prime = rho - 0.5*dt * ( vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
  vx_prime  = vx  - 0.5*dt * ( vx * vx_dx + vy * vx_dy + (1/rho) * P_dx )
  vy_prime  = vy  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + (1/rho) * P_dy )
  
  # extrapolate in space to face centers
  rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, dx)
  vx_XL,  vx_XR,  vx_YL,  vx_YR  = extrapolateInSpaceToFace(vx_prime,  vx_dx,  vx_dy,  dx)
  vy_XL,  vy_XR,  vy_YL,  vy_YR  = extrapolateInSpaceToFace(vy_prime,  vy_dx,  vy_dy,  dx)
  
  # compute fluxes (local Lax-Friedrichs/Rusanov)
  flux_Mass_X, flux_Momx_X, flux_Momy_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR)
  flux_Mass_Y, flux_Momy_Y, flux_Momx_Y = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR)
  
  # update solution
  Mass   = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
  Momx   = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
  Momy   = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
  
  # update time
  #t += dt

  return rho, vx, vy, Mass, Momx, Momy, dx, dt


@jit
def do_simulation(rho, vx, vy, dx, tEnd):
  """
    Run the finite volume simulation
  rho      is a matrix of the density field
  """

  Nt  = 300          # number of timesteps
  dt  = tEnd / Nt    # timestep  (use fixed timestep for simplicity)

  # Get conserved variables
  Mass, Momx, Momy = getConserved( rho, vx, vy, dx**2 )

  # Simulation Main Loop
  values = jax.lax.fori_loop(0, Nt, update_sim, init_val=(rho, vx, vy, Mass, Momx, Momy, dx, dt))
  rho, vx, vy, Mass, Momx, Momy, dx, dt = values
                    
  return rho


@jit 
def loss_function(x, rho, dx, tEnd, rho_target):
  vx, vy = x
  rho = do_simulation(rho, vx, vy, dx, tEnd)
  return jnp.mean( (rho - rho_target)**2 )



def main():
  """ Finite Volume simulation """

  # Simulation parameters
  N           = 100       # resolution           
  #courant_fac = 0.5       # Courant factor
  #t           = 0.0       # current time of the simulation
  tEnd        = 1.0       # time at which simulation ends

  # Mesh
  Lbox = 1.0
  dx = Lbox / N
  xlin = jnp.linspace(0.5*dx, Lbox-0.5*dx, N)
  X, Y = jnp.meshgrid( xlin, xlin )
  

  # Define the target density field from .png image
  rho_target = jnp.fliplr(jnp.array(img.imread('target.png')[:,:,0],dtype=float))
  rho_target = 1.0 + 0.02*(rho_target-0.5)
  # normalize so average density is 1
  rho_target /= jnp.mean(rho_target)

  # Now use autodiff to find initial conditions that generate the result

  rho = jnp.ones(X.shape)
  vx = jnp.zeros(X.shape)
  vy = jnp.zeros(X.shape)

  optimizer = ScipyMinimize(method="l-bfgs-b", fun=loss_function, tol=1e-8, options={'disp': True})

  sol = optimizer.run((vx, vy), rho, dx, tEnd, rho_target)


  # Carry out the simulation with the optimized initial conditions, and plot its time evolution
  fig = plt.figure(figsize=(4,4), dpi=100)
  cmap = plt.cm.bwr
  cmap.set_bad('LightGray')

  rho = jnp.ones(X.shape)
  vx = sol.params[0]
  vy = sol.params[1]
  Nt  = 300
  dt  = tEnd / Nt
  Mass, Momx, Momy = getConserved( rho, vx, vy, dx**2 )
  for i in range(Nt):
    values = (rho, vx, vy, Mass, Momx, Momy, dx, dt)
    rho, vx, vy, Mass, Momx, Momy, dx, dt = update_sim(i, values)
    # Make Plot
    plt.cla()
    plt.imshow(rho, cmap=cmap)
    plt.clim(.85,1.15)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  
    ax.set_aspect('equal')      
    plt.pause(0.001)

  # Save final figure
  plt.savefig('rho.png',dpi=240)
  plt.show()

  # plot the initial velocity field that gives rise to the result
  fig = plt.figure(figsize=(4,4), dpi=100)
  vnorm = jnp.sqrt(sol.params[0]**2 + sol.params[1]**2)
  plt.imshow(vnorm, cmap=cmap)
  plt.quiver(sol.params[0], sol.params[1])
  ax = plt.gca()
  ax.invert_yaxis()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)  
  ax.set_aspect('equal') 
  plt.savefig('vel0.png',dpi=240)
  plt.show()

  return 0



if __name__== "__main__":
  main()