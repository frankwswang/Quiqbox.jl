<p align="center">
    <a href="https://frankwswang.github.io/Quiqbox.jl/stable/">
        <img width="500" src="docs/src/assets/logo.png" alt="Quiqbox.jl">
    </a>
</p>

**Quiqbox** is a quantum chemistry and quantum physics software package that starts around Gaussian basis set optimization for electronic structure problems. Quiqbox is written in pure [Julia](https://julialang.org/). This work is supported by the U.S. Department of Energy under Award No. DESC0019374.

| Documentation | Code Status | License |
| :---: | :---: | :---: |
| [![][Doc-s-img]][Doc-stable] | [![codecov][codecov-img]][codecov-url] [![CI][GA-CI-img]][GA-CI-url] [![CI-JN][GA-CI-JN-img]][GA-CI-JN-url] [![][New-commits-img]][New-commits-url] | [![License: MIT][License-img]][License-url] |

<br />

# Features

* Native 1-electron and 2-electron integral functions.
* Floating and fixed-position contracted Gaussian-type orbital (CGTO).
* Linear combination of multi-center GTOs (MCGTO) as a basis function.
* Restricted (closed-shell) and unrestricted (open-shell) Hartree–Fock methods (RHF & UHF).
* Variational optimization of basis sets based on automatic differentiation (AD) and symbolic differentiation (SD).

# Setup

## Supported operating systems (64-bit)
* Linux
* Mac OS
* Windows

## Julia compatibility
Quiqbox will always try to support the [latest released version](https://julialang.org/downloads/#current_stable_release) of Julia as soon as possible. On the other hand, the backward compatibility for previous Julia versions is not guaranteed but can be checked [here](https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JS-older.yml).

## Installation in Julia [REPL](https://docs.julialang.org/en/v1/manual/getting-started/)

Type `]` in the default [Julian mode](https://docs.julialang.org/en/v1/stdlib/REPL/#The-Julian-mode) to switch to the [Pkg mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode):

```julia
(@v1.x) pkg>
```

Type the following command and hit *Enter* key to install Quiqbox:

```julia
(@v1.x) pkg> add Quiqbox
```

After the installation completes, hit *Backspace* key to go back to the Julian mode and use [`using`](https://docs.julialang.org/en/v1/base/base/#using) to load Quiqbox:

```julia
julia> using Quiqbox
```

# Showcase

## Combine atomic orbitals
```julia
points = GridBox((1,0,0), 1.4).point

bsH₂ = genBasisFunc.(points, "STO-3G") |> flatten
```

## Build a customized basis set
```julia
gf = GaussFunc(1.0, 0.75)

bs = genBasisFunc.(points, Ref(gf)) .+ bsH₂
```

## Run the Hartree-Fock method
```julia
nuc = ["H", "H"]

coords = coordOf.(points)

runHF(bs, nuc, coords)
```

## Optimize the basis set
```julia
pars = markParams!(bs, true)

optimizeParams!(pars, bs, nuc, coords)
```

# Documentation

For more information on the package, please read the [documentation of the latest release][Doc-stable]. For unreleased/experimental features, please refer to the [development documentation][Doc-latest].

<br />
<br />

<p align="center">
    <a href="https://jdwhitfield.com/">
        <img width=400 src="docs/src/assets/groupLogo.svg" alt="Whitfield Group">
    </a>
</p>

<br />

[Doc-stable]:  https://frankwswang.github.io/Quiqbox.jl/stable
[Doc-latest]:  https://frankwswang.github.io/Quiqbox.jl/dev
[Doc-s-img]:   https://img.shields.io/github/v/release/frankwswang/Quiqbox.jl?label=latest%20release
[Doc-l-img]:   https://img.shields.io/badge/docs-latest-blue.svg
[GA-CI-img]:   https://img.shields.io/github/workflow/status/frankwswang/Quiqbox.jl/CI-JS-latest?label=Julia%20latest
[GA-CI-url]:   https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JS-latest.yml
[GA-CI-JN-img]:https://img.shields.io/github/workflow/status/frankwswang/Quiqbox.jl/CI-JN?label=Julia%20nightly
[GA-CI-JN-url]:https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JN.yml
[codecov-img]: https://img.shields.io/codecov/c/github/frankwswang/Quiqbox.jl/main?label=coverage&token=Z1XOA39DV2
[New-commits-img]: https://img.shields.io/github/commits-since/frankwswang/Quiqbox.jl/latest?color=teal&include_prereleases
[New-commits-url]: https://github.com/frankwswang/Quiqbox.jl/commits/main
[codecov-url]: https://codecov.io/gh/frankwswang/Quiqbox.jl
[License-img]: https://img.shields.io/badge/MIT%20License-blueviolet.svg
[License-url]: https://github.com/frankwswang/Quiqbox.jl/blob/main/LICENSE