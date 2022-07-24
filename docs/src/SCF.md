# Self-Consistent Field Methods

## Hartree-Fock methods

Quiqbox supports basic Hartree-Fock methods with various configurations: 

| Items | Options |
| :---  |  ---:   |
| HF Types | restricted closed-shell (RHF), unrestricted open-shell (UHF) |
| Initial Guesses | core Hamiltonian, generalized Wolfsberg-Helmholtz, superposition of atomic densities (SAD), pre-defined coefficient matrix |
| Converging Methods | direct diagonalization, [direct inversion in the iterative subspace (DIIS)](https://onlinelibrary.wiley.com/doi/10.1002/jcc.540030413), [E-DIIS](https://aip.scitation.org/doi/abs/10.1063/1.1470195), [A-DIIS](https://aip.scitation.org/doi/10.1063/1.3304922), combinations of multiple methods |
| DIIS-type Method Solvers | Lagrange multiplier solver, [L-BFGS](https://github.com/JuliaNLSolvers/Optim.jl) solver |

### Basic Hartree-Fock

To run a Hartree-Fock method, the lines of code required in Quiqbox are as simple as below:
```@repl 2
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide
nuc = ["H", "H"];

nucCoords = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

bs = genBasisFunc.(nucCoords, "STO-3G", nuc) |> flatten

resRHF = runHF(bs, nuc, nucCoords)

@show resRHF.Ehf resRHF.C resRHF.Eo resRHF.occu;
```

After the SCF procedure, one can also store the result in a `MatterByHF` for further data processing such as generating a [Molden](@ref) file.
```@repl 2
mol = MatterByHF(resRHF); 
```

### Flexible core functions

If the user wants to fine-tune the SCF iteration to achieve better performance, Quiqbox has provided various core types and functions that allow the user to customize the HF methods:

[`HFconfig`](@ref)

[`SCFconfig`](@ref)

[`runHFcore`](@ref)

## Stand-alone integral functions

Quiqbox also provides efficient native functions for one-electron and two-electron integral calculations.

### One-electron functions

[`overlap`](@ref)

[`overlaps`](@ref)

[`eKinetic`](@ref)

[`eKinetics`](@ref)

[`neAttraction`](@ref)

[`neAttractions`](@ref)

[`coreHij`](@ref)

[`coreH`](@ref)

### Two-electron functions

[`eeInteraction`](@ref)

[`eeInteractions`](@ref)