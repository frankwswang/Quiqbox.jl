# Quiqbox.jl

[Quiqbox](https://github.com/frankwswang/Quiqbox.jl) is a quantum chemistry and quantum physics software package that starts around Gaussian basis set optimization for electronic structure problems. Quiqbox is written in pure [Julia](https://julialang.org/). This work is supported by the U.S. Department of Energy under Award No. DESC0019374.

## Features

* Native 1-electron and 2-electron integral functions.
* Floating and fixed-position contracted Gaussian-type orbital (CGTO).
* Mixed-contracted GTO (linear combination of GTOs with mixed centers or orbital angular momentum) as a basis function.
* Restricted (closed-shell) and unrestricted (open-shell) Hartree–Fock methods (RHF & UHF).
* Variational optimization of basis sets based on automatic differentiation (AD) and symbolic differentiation (SD).

## Setup

### OS (64-bit) support
* Generic Linux
* macOS
* Windows

**NOTE:** Each operating system (OS) platform is only tested on the x86-64 architecture. The support of those systems on different architectures (such as macOS on ARM architecture) is not guaranteed.

### Julia (64-bit) compatibility
Quiqbox will always try to support the [latest stable release](https://julialang.org/downloads/#current_stable_release) of 64-bit Julia as soon as possible. On the other hand, backward compatibility with previous versions is not guaranteed but can be checked [here](https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JS-older.yml).

### Installation in Julia [REPL](https://docs.julialang.org/en/v1/manual/getting-started/)

Type `]` in the default [Julian mode](https://docs.julialang.org/en/v1/stdlib/REPL/#The-Julian-mode) to switch to the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode):

```julia
(@v1.x) pkg>
```

Type the following command and hit *Enter* key to install Quiqbox:

```julia
(@v1.x) pkg> add Quiqbox
```

After the installation completes, hit the *Backspace* key to go back to the Julian mode and use [`using`](https://docs.julialang.org/en/v1/base/base/#using) to load Quiqbox:

```julia
julia> using Quiqbox
```

For more basic usage of the programming language behind Quiqbox, Julia, please refer to [the official documentation](https://docs.julialang.org/).


## Manual Contents
```@contents
Pages = ["basis.md", "SCF.md", "optimization.md"]
Depth = 3
```
