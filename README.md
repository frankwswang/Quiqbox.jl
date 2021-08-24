# Quiqbox.jl



[![CI][GA-CI-img]][GA-CI-url]
[![codecov][codecov-img]][codecov-url]
[![License: MIT][License-img]][License-url]

A Julia software package that aims for floating Gaussian-type orbital (GTO) optimization in electronic structure problems.

# Features

* Floating or fixed-basis GTO configuration.
* Symbolic representation of basis functions.
* Restricted closed-shell Hartree–Fock method (RHF).
* Unrestricted open-shell Hartree–Fock method (UHF).
* Molecular orbital data output in [Molden](https://www3.cmbi.umcn.nl/molden/) file format.
* Standalone 1-electron and 2-electron integral functions (powered by [libcint_jll](https://github.com/JuliaBinaryWrappers/libcint_jll.jl)).
* Variational orbital geometry optimization based on Automatic differentiation (AD).

[GA-CI-img]:   https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI.yml/badge.svg?branch=main
[GA-CI-url]:   https://github.com/frankwswang/Quiqbox.jl/actions/workflows/CI.yml
[codecov-img]: https://codecov.io/gh/frankwswang/Quiqbox.jl/branch/main/graph/badge.svg?token=Z1XOA39DV2
[codecov-url]: https://codecov.io/gh/frankwswang/Quiqbox.jl
[License-img]: https://img.shields.io/badge/License-MIT-blue.svg
[License-url]: https://opensource.org/licenses/MIT