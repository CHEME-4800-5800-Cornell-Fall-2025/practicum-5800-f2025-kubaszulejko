module Compute

using ..Types: MyClassicalHopfieldNetworkModel
using DataStructures: CircularBuffer
using LinearAlgebra: dot

# Export public API
export recover

"""
    recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, 
            true_energy::Float32; maxiterations::Int64=1000, 
            patience::Union{Int,Nothing}=nothing,
            miniterations_before_convergence::Union{Int,Nothing}=nothing) 
            -> Tuple{Dict{Int64,Array{Int32,1}}, Dict{Int64,Float32}}

Recover a memorized pattern from a corrupted initial state using asynchronous updates.

# Arguments
- `model::MyClassicalHopfieldNetworkModel`: The trained Hopfield network
- `sₒ::Array{Int32,1}`: Initial (corrupted) state vector
- `true_energy::Float32`: Energy of the target memorized pattern (for reference)
- `maxiterations::Int64`: Maximum number of update steps (default: 1000)
- `patience::Union{Int,Nothing}`: Consecutive identical states required for convergence
                                 (default: 5 or greater)
- `miniterations_before_convergence::Union{Int,Nothing}`: Minimum iterations before checking convergence
                                                         (default: equal to patience)

# Returns
- `frames::Dict{Int64,Array{Int32,1}}`: State at each iteration
- `energydictionary::Dict{Int64,Float32}`: Energy at each iteration

# Algorithm
Uses asynchronous updates: at each step, randomly select a neuron and update its state
based on the sign of its weighted input. This minimizes the network energy to find
stored patterns encoded as energy minima.
"""
function recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, 
                true_energy::Float32; maxiterations::Int64=1000,
                patience::Union{Int,Nothing}=nothing,
                miniterations_before_convergence::Union{Int,Nothing}=nothing)
    
    # Set defaults for patience
    if isnothing(patience)
        patience = max(5, length(sₒ) ÷ 100)  # At least 5 or 1% of network size
    end
    
    if isnothing(miniterations_before_convergence)
        miniterations_before_convergence = patience
    end
    
    # Extract model parameters
    W = model.W
    b = model.b
    memories = model.memories
    N = length(sₒ)  # Number of neurons
    K = size(memories, 2)  # Number of stored patterns
    
    # Initialize state and tracking dictionaries
    s = copy(sₒ)
    frames = Dict{Int64, Array{Int32,1}}()
    energydictionary = Dict{Int64, Float32}()
    
    # Store initial state and energy
    frames[0] = copy(s)
    E = compute_energy(s, W, b)
    energydictionary[0] = E
    
    # Initialize state history for convergence checking
    state_history = CircularBuffer{Array{Int32,1}}(patience)
    
    # Main iteration loop
    converged = false
    t = 0
    
    while !converged && t < maxiterations
        t += 1
        
        # Asynchronous update: randomly select a neuron
        i = rand(1:N)
        
        # Compute activation: sum of weighted inputs
        activation = sum(W[i, j] * s[j] for j ∈ 1:N) - b[i]
        
        # Update state using sign function
        s[i] = sign(activation) |> Int32
        if s[i] == 0  # Handle case where activation is exactly 0
            s[i] = Int32(1)
        end
        
        # Store state and energy
        frames[t] = copy(s)
        E = compute_energy(s, W, b)
        energydictionary[t] = E
        
        # Add to state history
        push!(state_history, copy(s))
        
        # Check convergence conditions
        if t >= miniterations_before_convergence
            
            # Condition 1: Check if state has stabilized (all history states identical)
            if length(state_history) == patience
                all_same = all(s_hist == state_history[1] for s_hist in state_history)
                if all_same
                    converged = true
                    break
                end
            end
            
            # Condition 2: Check if current state matches any stored memory
            for k ∈ 1:K
                memory = memories[:, k]
                if all(s .== memory)
                    converged = true
                    break
                end
            end
            
            # Condition 3: Check if energy equals true minimum
            if abs(E - true_energy) < 1e-6
                converged = true
                break
            end
        end
    end
    
    return frames, energydictionary
end

"""
    compute_energy(s::Array{Int32,1}, W::Array{Float32,2}, b::Array{Float32,1}) -> Float32

Compute the energy of a network state.

# Arguments
- `s::Array{Int32,1}`: Current network state
- `W::Array{Float32,2}`: Weight matrix
- `b::Array{Float32,1}`: Bias vector

# Returns
- `Float32`: Network energy E(s) = -0.5*s'Ws - b's
"""
function compute_energy(s::Array{Int32,1}, W::Array{Float32,2}, b::Array{Float32,1})
    E = -0.5 * dot(s, W * s) - dot(b, s)
    return Float32(E)
end

end  # module Compute
