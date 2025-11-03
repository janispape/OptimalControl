## Optimal Control

This repository contains Python implementations of optimal control problems — mathematical models
for systems whose evolution can be influenced by a control function. The goal is to find control 
strategies that minimize or maximize a given performance measure.

The repository includes numerical and analytical examples based on the theory presented in:
- [Evans — An Introduction to Mathematical Optimal Control Theory](https://math.berkeley.edu/~evans/control.course.pdf)
- [Grüne & Pannek — Numerical Methods for Nonlinear Optimal Control Problems](https://epub.uni-bayreuth.de/id/eprint/2001/1/Gruene_num_meth_nonlin_optimal_control_2013.pdf)

### Implemented Examples
- **RocketCar** – A simple, analytically solvable control system illustrating bang–bang control behavior.
- **Moon Lander** - A model for fuel-efficient spacecraft landing using discretization-based optimization.
- **Inventory Control** – A dynamic optimization problem minimizing storage and ordering costs under demand.
- **Bee Population** – A biological control model maximizing queen production using Pontryagin’s principle.
Each example provides a clear numerical illustration of key optimal control concepts.

A more detailed description on the general problem can be found [here](https://janispape.github.io/OptimalControl/)