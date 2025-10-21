# Optimal Control

This repository contains files, that implements optimal control problems.
Optimal control describes the following problem: For a function 
$f: \mathbb{R}^n \times \mathbb{R}^m \rightarrow \mathbb{R}^n$ and some 
called _control_ $\alpha:[0,\infty) \rightarrow \mathbb{R}^m$ we consider
an ODE:
$$\cases{\dot{x}(t) = f(x(t),\alpha(t))\\x(0) = x_0}$$
We want to find a control $\alpha$ in some set $\mathcal{A} \subset
\{\alpha:[0,\infty) \rightarrow \mathbb{R}^m | \ \alpha \text{ is measurable}
\}$ (often, $\mathcal{A}$ is just a restriction of the images of the $\alpha$),
such that, for some cost functions 
$r:\mathbb{R}^n \times \mathbb{R}^m \rightarrow \mathbb{R}$ and 
$g:\mathbb{R}^n \rightarrow \mathbb{R}$, we minimize the functional $P$, defined
as:
$$P[\alpha] = \int_0^T r(x_\alpha(t), \alpha(t)) dt + g(x(T))$$
($x_\alpha$ is the solution of the ODE, $T$ is allowed to be infinite). An
introduction to the theory of optimal control problems which also contains the
examples in this repository can be found here:
https://math.berkeley.edu/~evans/control.course.pdf \
The numerical methods implemented are reviewed here: \
https://epub.uni-bayreuth.de/id/eprint/2001/1/Gruene_num_meth_nonlin_optimal_control_2013.pdf

## Rocket car