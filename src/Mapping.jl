(ef::Evaluator)(args...) = ef.f(args...)

abstract type ChainOperator <: Function end

struct PointerFunc{F<:Function, T<:NonEmptyTuple{Union{Int, Memory{Int}}}} <: Function
    apply::F
    pointer::T
    sourceID::UInt
end

(f::PointerFunc)(input, par) = f.apply(input, getindex.(Ref(par), f.pointer)...)

const ParamOp = Union{PointerFunc, ChainOperator}

struct JoinParallel{C<:ParamOp, F<:ParamOp, J<:Function} <: ChainOperator
    chain::C
    apply::F
    joint::J
end

(f::JoinParallel)(input, par) = f.joint(f.chain(input, par), f.apply(input, par))

JoinParallel(joint::Function) =  (x, y)->JoinParallel(x, y, joint)

struct InsertOnward{C<:ParamOp, F<:ParamOp} <: ChainOperator
    chain::C
    apply::F
end

(f::InsertOnward)(input, par) = f.apply(f.chain(input, par), par)


struct InsertInward{C<:ParamOp, F<:ParamOp} <: ChainOperator
    chain::C
    apply::F
end

(f::InsertInward)(input, par) = f.chain(f.apply(input, par), par)


function evalFunc(func::F, r) where {F<:Function}
    fCore, pars = unpackFunc(func)
    fCore(r, map(obtain, pars))
end


unpackFunc(f::Function) = unpackFunc!(f, (uniqueParams∘getParams)(f))

unpackFunc!(f::Function, paramSet::PBoxCollection) = 
unpackFunc!(SelectTrait{ParameterStyle}()(f), f, paramSet)

unpackFunc!(::IsParamFunc, f::Function, paramSet::PBoxCollection) = 
unpackParamFunc!(f, paramSet)

function unpackFunc!(::NotParamFunc, f::Function, paramSet::PBoxCollection)
    params = getParams(f)
    ids = locateParam!(paramSet, params)
    fCopy = deepcopy(f)
    paramDoubles = deepcopy(params)
    foreach(p->setScreenLevel!(p.source, 1), paramDoubles)
    evalCore = function (x, pVals::Vararg)
        for (p, v) in zip(paramDoubles, pVals)
            setVal!(p, v)
        end
        fCopy(x)
    end
    PointerFunc(evalCore, ids, objectid(paramSet)), params
end