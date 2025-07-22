# Self-Consistent Field Methods

## Hartree–Fock methods

Quiqbox supports basic Hartree–Fock methods with various configurations: 

| Items | Options |
| :---  |  ---:   |
| HF Types | restricted closed-shell (RHF), unrestricted open-shell (UHF) |
| Initial Guesses | core Hamiltonian, generalized Wolfsberg-Helmholtz, superposition of atomic densities (SAD), pre-defined coefficient matrix |
| Converging Methods | direct diagonalization, [direct inversion in the iterative subspace (DIIS)](https://onlinelibrary.wiley.com/doi/10.1002/jcc.540030413), [E-DIIS](https://aip.scitation.org/doi/abs/10.1063/1.1470195), [A-DIIS](https://aip.scitation.org/doi/10.1063/1.3304922), combinations of multiple methods |
| DIIS-type Method Solvers | Lagrange multiplier solver, [L-BFGS](https://github.com/Gnimuc/LBFGSB.jl) solver |

### Basic Hartree–Fock

To run a Hartree–Fock method, the lines of code required in Quiqbox are as simple as below:
```@setup 2
    push!(LOAD_PATH,"../../src/")
    using Quiqbox
```
```@repl 2
nuc = [:H, :H];

nucCoords = [(-0.7, 0.0, 0.0), (0.7, 0.0, 0.0)];

bs = reduce(vcat, genGaussTypeOrbSeq.(nucCoords, nuc, "STO-3G"));

resRHF = runHartreeFock(NuclearCluster(nuc, nucCoords), bs);

@show resRHF.energy resRHF.coeff resRHF.occu;
```

### Flexible core functions

If the user wants to fine-tune the SCF iteration to achieve better performance, Quiqbox has provided various core types and functions that allow the user to customize the HF methods:

[`HFconfig`](@ref)

[`SCFconfig`](@ref)

## Stand-alone integral functions

Quiqbox also provides efficient native functions for one-electron and two-electron integral calculations.

### One-electron functions

[`overlap`](@ref)

[`overlaps`](@ref)

[`elecKinetic`](@ref)

[`elecKinetics`](@ref)

[`nucAttraction`](@ref)

[`nucAttractions`](@ref)

[`coreHamiltonian`](@ref)

### Two-electron functions

[`elecRepulsion`](@ref)

[`elecRepulsions`](@ref)