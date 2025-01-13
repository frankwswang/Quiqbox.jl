struct Identity <: DirectOperator end

(::Identity)(f::Function) = itself(f)