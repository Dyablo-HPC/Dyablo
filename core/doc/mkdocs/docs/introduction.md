# Solving compressible fluid flow equations on AMR grid

## About Euler equations

Just showing off by demoing we can write latex equations from markdown:

Let's define the vector of conservative variables $`U=(\rho, \rho u, \rho v, \rho w, E)`$ and the corresponding vector of primitive variables $`W=(\rho, u, v, w, p)`$, where $`\rho`$ is the fluid density, $`u,v,w`$ are the three cartesian components of the velocity vector field, $`E`$ is the total energy per unit volume and $`p`$ is the pressure.

Let $`c=\sqrt{\left(\frac{\partial p}{\partial \rho}\right)_s} = \sqrt{\frac{\gamma p}{\rho}}`$ be the speed of sound.

Then the 1d Euler system of equations in conservative form reads:

```math
\begin{array}{ccccc}
    \partial_t \rho & + & \partial_x(\rho u) & = & 0,\\
    \partial_t (\rho u) & + & \partial_x(\rho u^2+p) & = & 0,\\
    \partial_t E & + & \partial_x (u(E+p)) & = & 0,\\
  \end{array}
```
or in short notations:
```math
\partial_t \mathbf{U} + \partial_x \mathbf{F(U)} = \mathbf{0}
```

where the flux function is defined by
```math
\mathbf{F(U)} = \left [
  \begin{array}{c}
    \rho u \\
    \rho u^2 + p \\
    u (E + p)
  \end{array} \right]
```

- total (internal + kinetic) energy per unit volume $`E = \rho \left( e + \frac{1}{2} u^2 \right) = \frac{p}{\gamma-1} + \frac{1}{2} \rho u^2`$,
- specific (per mass unit) internal energy $`e=\frac{p}{(\gamma-1)\rho}`$.
- total enthalpy $`H = (E + p)/\rho`$ per mass unit (specific enthalpy)
- another useful relation $`\frac{\gamma E}{\rho}=H+(\gamma-1)u^2/2`$.

