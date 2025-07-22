using Aqua
using Quiqbox

@testset "Aqua.jl-Required Test" begin
    Aqua.test_all(Quiqbox)
end