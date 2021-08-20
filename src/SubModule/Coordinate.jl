module Coordinate

export intrRots, extrRots

using Rotations, StaticArrays

#===== Spatial operation functions =====#
function extrRots(vec::Union{SArray, Array{<:Real, 1}}, op::Rotation...; 
                  bodyFrameTransMat::Rotation=one(UnitQuaternion))
    rots = bodyFrameTransMat * (op |> reverse |> prod) * inv(bodyFrameTransMat)
    rots*vec, rots
end

function extrRots(vecs::Array{<:AbstractArray, 1}, op::Rotation...; 
                  bodyFrameTransMat::Rotation=one(UnitQuaternion))
    ans = [extrRots(i, op..., bodyFrameTransMat=bodyFrameTransMat) for i in vecs]
    ans .|> first, ans[1][2]
end


function intrRots(vec::Union{SArray, Array{<:Real, 1}}, op::Rotation...; 
                  bodyFrameTransMat::Rotation=one(UnitQuaternion))
    rots = bodyFrameTransMat * (op |> prod) * inv(bodyFrameTransMat)
    rots*vec, rots
end

function intrRots(vecs::Array{<:AbstractArray, 1}, op::Rotation...; 
                  bodyFrameTransMat::Rotation=one(UnitQuaternion))
    ans = [intrRots(i, op..., bodyFrameTransMat=bodyFrameTransMat) for i in vecs]
    ans .|> first, ans[1][2]
end

end