const QuiqboxSubModules = ["Coordinate", "Molden"]

# Initialization function.
function __init__()
    tryIncluding(QuiqboxSubModules...)
end


# Function for submudole loading and integrity checking.
function tryIncluding(subModuleName::String...)
    errorSubM = String[]
    errorInfo = []
    QuiqboxPath = @__DIR__
    for SMi in subModuleName
        try
            include(QuiqboxPath*"/SubModule/"*SMi*".jl")
        catch err
            push!(errorSubM, SMi)
            push!(errorInfo, err)
        end
    end
    if errorSubM |> !isempty
        l = length(errorSubM) 
        l == 1 ? s = "" : s = "s" 
        errInfo = ""
        errSubM = ""
        for i = 1:l
            errInfo *= "[`"*errorSubM[i]*"`]\n"*string(errorInfo[i])*"\n\n"
            errSubM *= " `"*errorSubM[i]*"`,"
        end
        errSubM = errSubM[1:end-1]
        warning = """
            Submodule$s$(errSubM) failed loading and won't be useable. 
            Please check if you have properly installed and configured the required packages/softwares/environments; 
            otherwise the subModule$s might have been broken.

            `///magenta///However, this issue DOES NOT affect the integrity of the main module.`
                
            More INFO on the LOADING ERROR:\n"""*errInfo
        printStyledInfo(warning, title="WARNING:\n", titleColor=:light_yellow)
    end
    return nothing
end