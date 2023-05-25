module NormalizingFlowsTutorials
    using Flux
    using InvertibleNetworks
    using LinearAlgebra
    using ProgressMeter

    import Flux:train!

    export sample_posterior
    export train!
    
    include("functions.jl")
end
