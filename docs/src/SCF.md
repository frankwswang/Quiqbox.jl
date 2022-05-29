# Self-Consistent Field Methods

## Hartree-Fock Methods

Quiqbox supports basic Hartree-Fock methods with various configurations: 

| Items | Options |
| :---  |  ---:   |
| HF Types | Restricted Closed-Shell (RHF), Unrestricted Open-Shell (UHF) |
| Initial Guesses | Core Hamiltonian, Generalized Wolfsberg-Helmholtz, Superposition of Atomic Densities (SAD), User-defined Coefficient Matrix |
| Converging Methods | Direct Diagonalization, [DIIS](https://onlinelibrary.wiley.com/doi/10.1002/jcc.540030413), [EDIIS](https://aip.scitation.org/doi/abs/10.1063/1.1470195), [ADIIS](https://aip.scitation.org/doi/10.1063/1.3304922), Combinations of Multiple Methods |
| DIIS-type Method Solvers | Lagrange Multiplier Solver, [L-BFGS-B](https://github.com/Gnimuc/LBFGSB.jl) Solver |

### Basic Hartree-Fock

To run a Hartree-Fock method, the lines of code required in Quiqbox are as simple as below:
```@repl 3
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide
nuc = ["H", "H"];

nucCoords = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

bs = genBasisFunc.(nucCoords, ("STO-3G", "H") |> Ref) |> flatten

resRHF = runHF(bs, nuc, nucCoords)

@show resRHF.Ehf resRHF.C resRHF.Emo resRHF.occu
```

After the SCF procedure, one can also easily store the result in a `Molecule` for further data processing such as generating [Molden](@ref) files.
```@repl 3
mol = Molecule(bs, nuc, nucCoords, resRHF);
```

### Flexible core functions

If the user wants to fine-tune part of the SCF iteration steps to achieve better performance, Quiqbox also has provided various more flexible core functions that 
allow the user to customize the HF methods:

[`HFconfig`](@ref)

[`SCFconfig`](@ref)

[`runHFcore`](@ref)

## Standalone Integral Functions

Quiqbox also provides several integral functions that can be used independently of any SCF functions if intended.

### One-electron functions

[`overlap`](@ref)

[`overlaps`](@ref)

[`elecKinetic`](@ref)

[`elecKinetics`](@ref)

[`nucAttraction`](@ref)

[`nucAttractions`](@ref)

[`coreHij`](@ref)

[`coreH`](@ref)

### Two-electron functions

[`eeInteraction`](@ref)

[`eeInteractions`](@ref)