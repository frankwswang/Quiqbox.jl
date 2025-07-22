using PrecompileTools: @setup_workload   as @prepareWorkload
using PrecompileTools: @compile_workload as @compileWorkload

function precompileField(field::FieldAmplitude{C, D}, ::Val{B}=Val(false)) where 
                        {T<:Real, C<:RealOrComplex{T}, D, B}
    centerInput = ntuple(_->zero(T), Val(D))

    field(centerInput)
    unpackFunc(field)

    if B
        primBasisMD = PrimitiveOrb(centerInput .- D, field)
        primBasisPD = PrimitiveOrb(centerInput .+ D, field)
        primBasisMN = PrimitiveOrb(primBasisMD, renormalize=true)
        primBasisPN = PrimitiveOrb(primBasisPD, renormalize=true)

        primBasisMD(centerInput)
        primBasisMN(centerInput)

        primBasesD = [primBasisMD, primBasisPD]
        primBasesN = [primBasisMN, primBasisPN]
        weights = T[2, 3]

        compBasisDD = CompositeOrb(primBasesD, weights)
        compBasisND = CompositeOrb(primBasesN, weights)
        compBasisNN = CompositeOrb(primBasesN, weights, renormalize=true)

        compBasisDD(centerInput)
        compBasisND(centerInput)
        compBasisNN(centerInput)

        if D==3
            bs = [primBasisMD, primBasisPD, primBasisMN, primBasisPN]
            coreHamiltonian([:H], [(0., 0., 0.)], bs)
            elecRepulsions(bs)
            overlaps(bs)
        end
    end

    nothing
end


@prepareWorkload begin
    fieldSeq1 = map(( (0,), (1, 1), (1, 1, 0) )) do ang
        PolyRadialFunc(GaussFunc(5.0), ang)
    end

    @compileWorkload begin
        for field in fieldSeq1
            precompileField(field, Val(true))
        end
    end
end