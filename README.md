<p align="center">
    <a href="https://frankwswang.github.io/Quiqbox.jl/stable/">
        <img width="500" src="docs/src/assets/logo.png" alt="Quiqbox.jl">
    </a>
</p>

**Quiqbox** is a quantum chemistry and quantum physics software package that starts off around Gaussian basis set optimization for electronic structure problems. Quiqbox is written in pure [Julia](https://julialang.org/). This work is supported by the U.S. Department of Energy under Award No. DESC0019374.

| Documentation | Code Status | License |
| :---: | :---: | :---: |
| [![][Doc-s-img]][Doc-stable] | [![CI][GA-CI-img]][GA-CI-url] [![codecov][codecov-img]][codecov-url] [![CI-JN][GA-CI-JN-img]][GA-CI-JN-url] | [![License: MIT][License-img]][License-url] |

<br />

# Features

* Floating and fixed-position Gaussian-type orbital (GTO).
* Linear combination of GTOs with different centers as a basis function.
* Symbolic representation of basis function expression.
* Standalone 1-electron and 2-electron integral functions.
* Restricted (closed-shell) and unrestricted (open-shell) Hartree–Fock methods (RHF & UHF).
* Molecular orbital data output in [Molden](https://www3.cmbi.umcn.nl/molden/) file format.
* Variational optimization of orbital parameters based on automatic differentiation (AD).

# Setup

## Supported system platforms (64-bit)
* Linux
* Mac OS
* Windows

## Julia Environment
* [1.6+](https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JS-older.yml)

## Installation in Julia [REPL](https://docs.julialang.org/en/v1/manual/getting-started/)

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

# Showcase

## Apply existed basis set
```julia
coords = [[-0.7,0,0], [0.7,0,0]]

bsH₂ = genBasisFunc.(coords, "STO-3G") |> flatten
```

## Build your own basis set
```julia
bs = genBasisFunc.(coords, fill(GaussFunc(1, 0.75), 2))
```

## Run Hartree-Fock method
```julia
nuc = ["H", "H"]

runHF(bs, nuc, coords)
```

## Optimize a basis set
```julia
pars = markParams!(bs, true)

optimizeParams!(pars[end-1:end], bs, nuc, coords)
```

# Documentation

For more information on how to use the package, please read the [documentation of released versions][Doc-stable]. For unreleased/experimental features, please refer to the [latest documentation][Doc-latest].

To learn more about the basic usage of the programming language behind Quiqbox, **Julia**, [the official documentation](https://docs.julialang.org/) or [this official tutorial](https://juliaacademy.com/p/intro-to-julia) is recommended.

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
[Doc-s-img]:   https://img.shields.io/github/v/release/frankwswang/Quiqbox.jl?label=Latest%20release
[Doc-l-img]:   https://img.shields.io/badge/docs-latest-blue.svg
[GA-CI-img]:   https://img.shields.io/github/workflow/status/frankwswang/Quiqbox.jl/CI?label=Julia%20latest
[GA-CI-url]:   https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JS-latest.yml
[GA-CI-JN-img]:https://img.shields.io/github/workflow/status/frankwswang/Quiqbox.jl/CI-JN?label=Julia%20nightly
[GA-CI-JN-url]:https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JN.yml
[codecov-img]: https://img.shields.io/codecov/c/github/frankwswang/Quiqbox.jl/main?label=Coverage&token=Z1XOA39DV2
[codecov-url]: https://codecov.io/gh/frankwswang/Quiqbox.jl
[License-img]: https://img.shields.io/badge/License-MIT-blueviolet.svg
[License-url]: https://github.com/frankwswang/Quiqbox.jl/blob/main/LICENSE