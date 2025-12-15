# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_IMAGES = joinpath(_ROOT, "images-uncorrelated");

# load external packages -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using Images
using ImageInTerminal
using FileIO
using ImageIO
using OneHotArrays
using Statistics
using JLD2
using LinearAlgebra
using Plots
using Colors
using Distances
using DataStructures
using Test
using IJulia

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));

# expose module symbols to Main from a single place to avoid conflicting import warnings when
# the file is included multiple times in the same Julia session (e.g., re-running notebook cells).
if !isdefined(Main, :MyClassicalHopfieldNetworkModel)
    using .Types: MyClassicalHopfieldNetworkModel, decode, hamming
end
if !isdefined(Main, :build)
    using .Factory: build
end
if !isdefined(Main, :recover)
    using .Compute: recover
end

# Helper: force reload of source modules and rebind exported symbols into Main.
# Use this inside running notebook kernels to pick up edits on disk.
function reload_sources()
    include(joinpath(_PATH_TO_SRC, "Types.jl"))
    include(joinpath(_PATH_TO_SRC, "Factory.jl"))
    include(joinpath(_PATH_TO_SRC, "Compute.jl"))

    # Rebind exported symbols into Main (do top-level using via eval so it's valid
    # when this helper is called from a running notebook kernel)
    # use Core.eval which accepts (Module, Expr) to perform top-level using in Main
    Core.eval(Main, :(using .Types: MyClassicalHopfieldNetworkModel, decode, hamming))
    Core.eval(Main, :(using .Factory: build))
    Core.eval(Main, :(using .Compute: recover))

    println("reload_sources: modules reloaded and symbols bound to Main")
    return nothing
end