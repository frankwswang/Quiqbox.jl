# Quiqbox.jl

**Quiqbox** is a quantum chemistry and quantum physics software package that starts off around Gaussian basis set optimization for electronic structure problems. Quiqbox is written in pure [Julia](https://julialang.org/).

## Features

* Floating and fixed-position Gaussian-type orbital (GTO).
* Linear combination of GTOs with different centers as a basis function.
* Symbolic representation of basis function expression.
* Standalone 1-electron and 2-electron integral functions.
* Restricted (closed-shell) and unrestricted (open-shell) Hartree–Fock methods (RHF & UHF).
* Molecular orbital data output in [Molden](https://www3.cmbi.umcn.nl/molden/) file format.
* Variational optimization of orbital parameters based on automatic differentiation (AD).

## Setup

### Supported system platforms (64-bit)

* Linux
* Mac OS
* Windows

### Julia Environment

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

For more basic usage of the programming language behind Quiqbox, **Julia**, please refer to [the official documentation](https://docs.julialang.org/) or [this official tutorial](https://juliaacademy.com/p/intro-to-julia).
