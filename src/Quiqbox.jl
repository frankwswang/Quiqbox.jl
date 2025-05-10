module Quiqbox

include("Types.jl")
include("Dispatch.jl")

include("Lexicons.jl")
include("../lib/BasisSets/BasisSets.jl")

include("Tools.jl")
include("StringIO.jl")
include("Exception.jl")
include("ParallelUtils.jl")

include("Query.jl")
include("Collections.jl")
include("Mapping.jl")
include("Operators.jl")
include("Parameters.jl")
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

include("Differentiation/Finite.jl")

include("Integration/Samplers.jl")
include("Integration/Framework.jl")
include("Integration/Engines/Numerical.jl")
include("Integration/Engines/GaussianOrbitals.jl")
include("Integration/Interface.jl")

# include("Integrals/OneBody.jl")
# include("Integrals/TwoBody.jl")

end

## Development comment guideline
#!! Urgent issue
#!!... Urgent issue that's being implemented/fixed
#! Non-urgent issue
#? Consideration for change
#+ Potential Feature implementation

## Footnote comment template
#= Additional Method =#