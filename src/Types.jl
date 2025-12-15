module Types

# Export public API
export MyClassicalHopfieldNetworkModel, decode, hamming

# Type definition for Classical Hopfield Network Model
"""
    MyClassicalHopfieldNetworkModel

A classical Hopfield network model that stores and retrieves binary patterns.

# Fields
- `W::Array{Float32,2}`: Weight matrix (N × N), symmetric with zero diagonal
- `b::Array{Float32,1}`: Bias vector (N,), typically zero for classical Hopfield
- `energy::Dict{Int,Float32}`: Energy values of memorized patterns
- `memories::Array{Int32,2}`: Stored memory patterns (N × K), where K is number of patterns
"""
mutable struct MyClassicalHopfieldNetworkModel
    W::Array{Float32,2}
    b::Array{Float32,1}
    energy::Dict{Int,Float32}
    memories::Array{Int32,2}
end

"""
    decode(s::Array{Int32,1}) -> Array{Float32,2}

Convert a binary state vector (with values ±1) to a 28×28 image for visualization.

# Arguments
- `s::Array{Int32,1}`: Binary state vector of length 784 (28² pixels)

# Returns
- `Array{Float32,2}`: 28×28 image array with values in [0, 1]
"""
function decode(s::Array{Int32,1})
    N = length(s)
    n = Int(sqrt(N))
    
    # Create image array
    img = Array{Float32,2}(undef, n, n)
    
    # Convert binary states to pixel values
    # States of -1 map to 0 (black), states of +1 map to 1 (white)
    linearindex = 1
    for row ∈ 1:n
        for col ∈ 1:n
            if s[linearindex] > 0
                img[row, col] = 1.0
            else
                img[row, col] = 0.0
            end
            linearindex += 1
        end
    end
    
    return img
end

"""
    hamming(s1::Array{Int32,1}, s2::Array{Int32,1}) -> Int

Compute the Hamming distance between two binary state vectors.

# Arguments
- `s1::Array{Int32,1}`: First binary state vector
- `s2::Array{Int32,1}`: Second binary state vector

# Returns
- `Int`: Number of differing positions
"""
function hamming(s1::Array{Int32,1}, s2::Array{Int32,1})
    return sum(s1 .!= s2)
end

end  # module Types