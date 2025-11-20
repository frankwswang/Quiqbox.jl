# Quiqbox.jl

[Quiqbox](https://github.com/frankwswang/Quiqbox.jl) is a quantum chemistry and quantum physics software package that starts around Gaussian basis set optimization for electronic structure problems. Quiqbox is written in pure [Julia](https://julialang.org/).

## Features

* Support constructing floating and fixed-position contracted orbitals.
* Support constructing mixed-contracted Gaussian-type orbitals and building hybrid basis sets.
* Provide native one-electron and two-electron integral functions.
* Provide restricted (closed-shell) and unrestricted (open-shell) Hartree–Fock methods (RHF & UHF).
* Provide dynamic computation-graph based function generation and variational optimization.

## Setup

### OS and hardware platform support

* Windows (x86-64)
* Generic Linux (x86-64)
* macOS (x86-64 and Apple silicon)

### Julia (64-bit) compatibility

Quiqbox aims to support the [**current stable release** of 64-bit Julia](https://julialang.org/downloads/#current_stable_release) as soon as possible. The latest compatibility tests can be found [here](https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-Release.yml). Backward compatibility with previous Julia releases is not guaranteed but can also be checked [here](https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-Backward.yml).

### Installation in Julia [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)

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

## Citation

If you use Quiqbox in your research, please cite the following paper:

- Wang, W., & Whitfield, J. D. (2023). Basis set generation and optimization in the NISQ era with Quiqbox.jl. [*Journal of Chemical Theory and Computation, 19*(22), 8032-8052.](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00011)

## Documentation Contents

```@contents
Pages = ["SCF.md"]
Depth = 2
```