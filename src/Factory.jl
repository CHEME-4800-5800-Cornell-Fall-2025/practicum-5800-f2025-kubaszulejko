module Factory

using ..Types: MyClassicalHopfieldNetworkModel
using LinearAlgebra: dot

# Export public API
export build

"""
    build(::Type{MyClassicalHopfieldNetworkModel}, config::NamedTuple) -> MyClassicalHopfieldNetworkModel

Build a classical Hopfield network model using the Hebbian learning rule.

# Arguments
- `config::NamedTuple`: Configuration with field:
  - `memories::Array{Int32,2}`: Binary patterns to memorize (N × K), where N is number of neurons and K is number of patterns

# Returns
- `MyClassicalHopfieldNetworkModel`: Initialized Hopfield network with:
  - `W`: Weight matrix computed via Hebbian rule: W = (1/K) * Σ(s_i ⊗ s_i')
  - `b`: Bias vector (all zeros)
  - `energy`: Dictionary with energy of each memorized pattern
  - `memories`: Stored patterns

# Details
The Hebbian learning rule encodes K binary patterns as an outer product sum,
scaled by 1/K. The weight matrix is symmetric with zero diagonal (no self-connections).
"""
function build(::Type{MyClassicalHopfieldNetworkModel}, config::NamedTuple)
    
    # Extract memories
    memories = config.memories
    N, K = size(memories)  # N = number of neurons, K = number of patterns
    
    # Initialize weight matrix and bias
    W = zeros(Float32, N, N)
    b = zeros(Float32, N)
    
    # Compute weights using Hebbian learning rule
    # W = (1/K) * Σ(s_i ⊗ s_i'), where ⊗ is outer product
    for k ∈ 1:K
        s = memories[:, k]  # k-th pattern
        W += (s * s') / K   # Add outer product, normalized by K
    end
    
    # Zero out diagonal (no self-connections in Hopfield networks)
    for i ∈ 1:N
        W[i, i] = 0.0
    end
    
    # Compute energy of each stored pattern
    energy_dict = Dict{Int, Float32}()
    for k ∈ 1:K
        s = memories[:, k]
        E = -0.5 * dot(s, W * s) - dot(b, s)  # Energy: E(s) = -0.5*s'Ws - b's
        energy_dict[k] = Float32(E)
    end
    
    # Create and return model instance
    return MyClassicalHopfieldNetworkModel(
        W,
        b,
        energy_dict,
        memories
    )
end

end  # module Factory
