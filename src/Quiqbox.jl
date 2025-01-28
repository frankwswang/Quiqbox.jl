module Quiqbox

include("Types.jl")
include("Traits.jl")
include("Layout.jl")

include("Tools.jl")
# include("Formulae.jl")
include("StringIO.jl")
include("Exception.jl")
include("ParallelUtils.jl")

include("../lib/BasisSets/BasisSets.jl")

# include("Library.jl")
include("Lexicons.jl")
include("Parameters.jl")
include("Mapping.jl")
include("Graphs.jl")

include("Angular.jl")
include("Spatial.jl")
include("SpatialBasis.jl")
# include("HartreeFock.jl")
# include("Differentiation.jl")
# include("Box.jl")
# include("Optimization.jl")
# include("Matter.jl")
# include("Overload.jl")

include("Integrals/Operators.jl")
include("Integrals/Engines/Numerical.jl")
include("Integrals/Engines/GaussianOrbitals.jl")
include("Integrals/Framework.jl")
include("Integrals/Interface.jl")

# include("Integrals/OneBody.jl")
# include("Integrals/TwoBody.jl")

include("Behaviors.jl")

include("Initialization.jl")
end

## Development comment guideline
#!! Urgent issue
#!!... Urgent issue that's being implemented/fixed
#! Non-urgent issue
#? Consideration for change
#+ Potential Feature implementation

## Footnote comment template
#= Additional Method =#