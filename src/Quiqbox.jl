module Quiqbox

include("Types.jl")
include("Dispatch.jl")

include("../lib/BasisSets/BasisSets.jl")
include("Lexicons.jl")

include("Tools.jl")
include("Strings.jl")
include("Exception.jl")
include("ParallelUtils.jl")

include("Query.jl")
include("Caches.jl")
include("Iteration.jl")
include("Collections.jl")
include("Mapping.jl")
include("Operators.jl")
include("Parameters.jl")
include("Graphs.jl")
include("Arithmetic.jl")

include("Angular.jl")
include("Particles.jl")

include("FieldFunctions.jl")
include("OrbitalBases.jl")

include("Differentiation/Finite.jl")

include("Integration/Samplers.jl")
include("Integration/Framework.jl")
include("Integration/Engines/Numerical.jl")
include("Integration/Engines/BoysFunction.jl")
include("Integration/Engines/GaussianOrbitals.jl")
include("Integration/Interface.jl")

include("HartreeFock.jl")

include("IO.jl")

include("Precompilation.jl")

end

## Development comment style guide
#!! Urgent issue
#! Non-urgent issue
#? Consideration for change or new implementation
#> Code explanation
#- Code method reference

## Footnote comment template
#= Additional Method =#

## Variable naming style guide
#> functions and variables: `oneTT`, `OneTThree`, `oneTwoThree`
#> functions that always modify their inputs: `oneTwo!`
#> function arguments that may be modified: `name!Self`
#> Types (including constant type aliases): `OneTT`, `OneTThree`, `OneTwoThree`
#> Global constant variables: `CONSTVAR!!NameOfTheVariable`