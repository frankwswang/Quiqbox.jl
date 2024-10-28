(po::ParamOperator)(input, param) = evalFunc(po, input, param)


struct PointerFunc{F<:Function, T<:NonEmptyTuple{Union{Int, Memory{Int}}}} <: ParamOperator
    apply::F
    pointer::T
    sourceID::UInt
end

evalFunc(f::PointerFunc, input, param) = f.apply(input, getindex.(Ref(param), f.pointer)...)


struct OnlyInput{F<:Function} <: Evaluator{F}
    f::F
end

evalFunc(f::OnlyInput, input, _) = f.f(input)


struct OnlyParam{F<:Function} <: Evaluator{F}
    f::F
end

evalFunc(f::OnlyParam, _, param) = f.f(param)


struct ParamFunc{F<:Function} <: Evaluator{F}
    f::F
end

evalFunc(f::ParamFunc, input, param) = f.f(input, param)


struct JoinParallel{C<:ParamOperator, F<:ParamOperator, J<:Function} <: ChainOperator
    chain::C
    apply::F
    joint::J
end

const AddParallel{L, R} = JoinParallel{L, R, typeof(+)}
const MulParallel{L, R} = JoinParallel{L, R, typeof(*)}

evalFunc(f::JoinParallel, input, param) = 
f.joint(f.chain(input, param), f.apply(input, param))

JoinParallel(joint::Function) =  (x, y)->JoinParallel(x, y, joint)

struct InsertOnward{C<:ParamOperator, F<:ParamOperator} <: ChainOperator
    chain::C
    apply::F
end

evalFunc(f::InsertOnward, input, param) = f.apply(f.chain(input, param), param)


struct InsertInward{C<:ParamOperator, F<:ParamOperator} <: ChainOperator
    chain::C
    apply::F
end

evalFunc(f::InsertInward, input, param) = f.chain(f.apply(input, param), param)


function evalFunc(func::F, r) where {F<:Function}
    fCore, pars = unpackFunc(func)
    fCore(r, map(obtain, pars))
end


unpackFunc(f::Function) = unpackFunc!(f, ParamBox[])

unpackFunc!(f::Function, paramSet::PBoxAbtArray) = 
unpackFunc!(SelectTrait{ParameterStyle}()(f), f, paramSet)

unpackFunc!(::IsParamFunc, f::Function, paramSet::PBoxAbtArray) = 
unpackParamFunc!(f, paramSet)

function unpackFunc!(::NotParamFunc, f::Function, paramSet::PBoxAbtArray)
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