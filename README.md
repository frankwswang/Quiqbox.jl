<p align="center">
    <a href="https://frankwswang.github.io/Quiqbox.jl/stable/">
        <img width="500" src="docs/src/assets/logo.png" alt="Quiqbox.jl">
    </a>
</p>

**Quiqbox** is a quantum chemistry and quantum physics software package that starts around Gaussian basis set optimization for electronic structure problems. Quiqbox is written in pure [Julia](https://julialang.org/). This work is supported by the U.S. Department of Energy under Award No. DESC0019374.

| Documentation | Digital Object Identifier | Paper | License |
| :---: | :---: | :---: | :---: |
| [![][Doc-l-img]][Doc-latest] | [![][Zenodo-DOI-img]][Zenodo-DOI-url] |[![][arXiv-img]][arXiv-url] | [![License: MIT][License-img]][License-url] |



| Development Status |
|:---:|
| [![codecov][codecov-img]][codecov-url] [![CI][GA-CI-img]][GA-CI-url] [![][New-commits-img]][New-commits-url] |


<br />

# Features

* Native 1-electron and 2-electron integral functions.
* Floating and fixed-position contracted Gaussian-type orbital (CGTO).
* Mixed-contracted GTO (linear combination of GTOs with mixed centers or orbital angular momentum) as a basis function.
* Restricted (closed-shell) and unrestricted (open-shell) Hartree–Fock methods (RHF & UHF).
* Variational optimization of basis set parameters based on a hybrid analytical differentiation design combining automatic differentiation (AD) and symbolic differentiation (SD).

# Setup

## OS (64-bit) support
* Generic Linux
* macOS
* Windows

**NOTE:** Each operating system (OS) platform is only tested on the x86-64 architecture. The support of those systems on different architectures (such as macOS on ARM architecture) is not guaranteed.

## Julia (64-bit) compatibility
Quiqbox will always try to support the [latest stable release](https://julialang.org/downloads/#current_stable_release) of 64-bit Julia as soon as possible. On the other hand, backward compatibility with previous versions is not guaranteed but can be checked [here](https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-JS-older.yml).

## Installation in Julia [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)

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

# Showcase

## Combine atomic orbitals
```julia
points = GridBox((1,0,0), 1.4).point

bsH₂ = vcat(genBasisFunc.(points, "STO-3G")...)
```

## Build a customized basis set
```julia
gf = GaussFunc(1.0, 0.75)

bs = genBasisFunc.(points, Ref(gf)) .+ bsH₂
```

## Run the Hartree–Fock method
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
Objects defined by Quiqbox that are directly exported to the user have the corresponding docstring, which can be accessed through the [Help mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode) in Julia REPL. The [latest release's documentation][Doc-latest] contains all the docstrings and additional tutorials of the package. For unreleased/experimental features, please refer to the [developer documentation][Doc-dev].

# Citation
If you use Quiqbox in your research, please cite the following paper:
- [Wang, W., & Whitfield, J. D. (2023). Basis set generation and optimization in the NISQ era with Quiqbox.jl. *Journal of Chemical Theory and Computation, 19*(22), 8032-8052.][JCTC-url]

<br />
<br />

<p align="center">
    <a href="https://jdwhitfield.com/">
        <img width=450 src="docs/src/assets/groupLogo.png" alt="Whitfield Group">
    </a>
</p>

<br />

[Doc-l-img]:   https://img.shields.io/github/v/release/frankwswang/Quiqbox.jl?label=latest%20release&color=seagreen
[Doc-latest]:  https://frankwswang.github.io/Quiqbox.jl/stable
[Doc-dev]:  https://frankwswang.github.io/Quiqbox.jl/dev

[GA-CI-img]:   https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-Rel-Latest.yml/badge.svg
[GA-CI-url]:   https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI-Rel-Latest.yml

[codecov-img]: https://codecov.io/gh/frankwswang/Quiqbox.jl/branch/main/graph/badge.svg?token=Z1XOA39DV2
[codecov-url]: https://codecov.io/gh/frankwswang/Quiqbox.jl

[New-commits-img]: https://img.shields.io/github/commits-since/frankwswang/Quiqbox.jl/latest?color=teal&include_prereleases
[New-commits-url]: https://github.com/frankwswang/Quiqbox.jl/commits/main

[Zenodo-DOI-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.7448313.svg
[Zenodo-DOI-url]: https://doi.org/10.5281/zenodo.7448313

[arXiv-img]: https://img.shields.io/badge/arXiv-2212.04586-b31b1b.svg
[arXiv-url]: https://arxiv.org/abs/
[JCTC-url]: https://pubs.acs.org/doi/10.1021/acs.jctc.3c00011

[License-img]: https://img.shields.io/badge/License-MIT-yellow.svg
[License-url]: https://github.com/frankwswang/Quiqbox.jl/blob/main/LICENSE
