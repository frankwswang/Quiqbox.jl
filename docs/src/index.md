# Quiqbox.jl

**Quiqbox** is a quantum chemistry and quantum physics software package that starts around Gaussian basis set optimization for electronic structure problems. Quiqbox is written in pure [Julia](https://julialang.org/). This work is supported by the U.S. Department of Energy under Award No. DESC0019374.

## Features

* Native 1-electron and 2-electron integral functions.
* Floating and fixed-position contracted Gaussian-type orbital (CGTO).
* Linear combination of multi-center GTOs (MCGTO) as a basis function.
* Restricted (closed-shell) and unrestricted (open-shell) Hartree–Fock methods (RHF & UHF).
* Variational optimization of orbital parameters based on automatic differentiation (AD).

## Setup

### Supported system platforms (64-bit)

* Linux
* Mac OS
* Windows

### Julia environment

* [1.6+](https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI.yml)

### Installation in Julia [REPL](https://docs.julialang.org/en/v1/manual/getting-started/)

Type `]` to enter the [`Pkg` mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode):

```julia
(@v1.x) pkg>
```

Type `add Quiqbox` and hit *Enter* key to install Quiqbox:

```julia
(@v1.x) pkg> add Quiqbox
```

After the installation completes, hit *Backspace* key to go back to Julia REPL and use [`using`](https://docs.julialang.org/en/v1/base/base/#using) to load Quiqbox:

```julia
julia> using Quiqbox
```

For more basic usage of the programming language behind Quiqbox, **Julia**, please refer to [the official documentation](https://docs.julialang.org/).


## Manual Contents
```@contents
Pages = ["basis.md", "SCF.md", "optimization.md"]
Depth = 3
```
