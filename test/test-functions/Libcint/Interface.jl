"""

    arrayAlloc(arrayLength::Int, 
               anExampleOrType::Union{T, Type{T}}) where {T<:Real} -> 
    Ptr{T}

Allocate the memory for an array of specified length and element type, then return the 
pointer `Ptr` to it.
"""
function arrayAlloc(arrayLength::Int, elementType::Type{T}) where {T<:Real}
    memoryLen = arrayLength*sizeof(elementType) |> Cint
    ccall(:malloc, Ptr{T}, (Cint,), memoryLen)
end

arrayAlloc(arrayLength::Int, NumberExample::T) where {T<:Real} = 
arrayAlloc(arrayLength, typeof(NumberExample))


"""

    ArrayPointer{T, N} <: Any

Stores a pointer to the actual address of an array.

≡≡≡ Field(s) ≡≡≡

`ptr::Ptr{T}`: Pointer pointing to the memory address of the first element of the array.

`arr::Array{T, N}`: The mutable array linked to the pointer. As long as the pointer 
                    (memory) is not freed, the array is safely preserved.

≡≡≡ Initialization Method(s) ≡≡≡

    ArrayPointer(arr::Array{<:Real, N}, 
                 showReminder::Bool=true) where {N} -> ArrayPointer{T, N}

Create a `ArrayPointer` that contains a `Ptr` pointing to the actual memory address of the 
(first element of the) `Array`.

To avoid memory leaking, the user should use `free(x.ptr)` after the usage of 
`x::ArrayPointer` to free the occupied memory.

If `showReminder=true`, the constructor will pop up a message to remind the user of 
such operation.

**WARNING: This function might be completely removed in the future release.**
"""
struct ArrayPointer{T, N} <: Any
    ptr::Ptr{T}
    arr::Array{T, N}

    function ArrayPointer(arr::Array{<:Real, N}, showReminder::Bool=true) where {N}
        len = length(arr)
        elt =  eltype(arr)
        ptr = arrayAlloc(len, elt)
        unsafe_copyto!(ptr, pointer(arr |> copy), len)
        arr2 = unsafe_wrap(Array, ptr, size(arr))
        showReminder && Quiqbox.printStyledInfo("""
            Generating a C-array pointer-like object x`::ArrayPointer{$(elt)}`...
            Remember to use free(x.ptr) afterwards to prevent potential memory leaking.
            """)
        new{elt, N}(ptr, arr2)
    end
end


function cintFunc!(libcintFunc::Val, 
                   buf::Array{Float64, N}, shls::Vector{<:Signed}, atm::Vector{<:Signed}, 
                   natm::Signed, bas::Vector{<:Signed}, nbas::Signed, env::Vector{Float64}, 
                   opt::Ptr=C_NULL) where {N}
    shls = shls .|> Cint
    atm = atm .|> Cint
    natm = natm |> Cint
    bas = bas .|> Cint
    nbas = nbas |> Cint
    bufAP = ArrayPointer(buf, false)
    shlsAP = ArrayPointer(shls, false)
    atmAP = ArrayPointer(atm, false)
    basAP = ArrayPointer(bas, false)
    envAP = ArrayPointer(env, false)
    intPtrFunc!(libcintFunc, 
                bufAP.ptr, shlsAP.ptr, atmAP.ptr, natm, basAP.ptr, nbas, envAP.ptr, opt)
    copyto!(buf, bufAP.arr)
    Libc.free(bufAP.ptr)
    Libc.free(shlsAP.ptr)
    Libc.free(atmAP.ptr)
    Libc.free(basAP.ptr)
    Libc.free(envAP.ptr)
    buf
end