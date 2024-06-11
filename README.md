# finitevolume-jax
Compressible Euler equations solved with finite volume implemented in JAX, plugged into an optimization loop


### Philip Mocz (2024) Flatiron Institute, [@PMocz](https://twitter.com/PMocz)

Run the code in the command line with:

```console
python finitevolume.py
```

The code finds velocity field initial conditions to the compressible Euler equations (isothermal) that lead to a prescribed density field at t=1:

![Simulation](./rho.png)

The initial velocity field that gives rise to the flow is:

![InitialCondition](./vel0.png)