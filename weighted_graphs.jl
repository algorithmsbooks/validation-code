"""
Modification of the SimpleWeightedGraphs.jl package to allow for weighted graphs with abritrary states.
Used in the Algorithms for Validation book by Mykel J. Kochenderfer, Sydney M. Katz, Anthony L. Corso, and Robert J. Moss
"""

@with_kw struct WeightedGraph{T}
    g::SimpleWeightedDiGraph
    states2ind::Dict{T,Int} = Dict()
    ind2states::Dict{Int,T} = Dict()
end

function WeightedGraph{T}(nv::Int) where T
    return WeightedGraph{T}(SimpleWeightedDiGraph(nv), Dict(), Dict())
end

function WeightedGraph(vertices::Vector)
    nv = length(vertices)
    T = typeof(first(vertices))
    return WeightedGraph{T}(nv)
end

function SimpleWeightedGraphs.add_edge!(g::WeightedGraph{T}, si::T, sj::T, w::Real) where T
    val = values(g.states2ind)
    n = isempty(val) ? 0 : maximum(val)

    if !haskey(g.states2ind, si)
        n += 1
        g.states2ind[si] = n
    end
    i = g.states2ind[si]
    g.ind2states[i] = si

    if !haskey(g.states2ind, sj)
        n += 1
        g.states2ind[sj] = n
    end
    j = g.states2ind[sj]
    g.ind2states[j] = sj

    return add_edge!(g.g, i, j, w)    
end

function SimpleWeightedGraphs.outneighbors(g::WeightedGraph{T}, si::T) where T
    i = g.states2ind[si]
    J = outneighbors(g.g, i)
    return map(j->g.ind2states[j], J)
end

function SimpleWeightedGraphs.inneighbors(g::WeightedGraph{T}, si::T) where T
    i = g.states2ind[si]
    J = inneighbors(g.g, i)
    return map(j->g.ind2states[j], J)
end

function SimpleWeightedGraphs.edges(g::WeightedGraph{T}) where T
    E = []
    for edge in edges(g.g)
        (i,j,w) = Tuple(edge)
        si = g.ind2states[i]
        sj = g.ind2states[j]
        push!(E, (si, sj, w))
    end
    return E
end

function SimpleWeightedGraphs.get_weight(g::WeightedGraph{T}, si::T, sj::T) where T
    i = g.states2ind[si]
    j = g.states2ind[sj]
    return get_weight(g.g, i, j)
end

function to_matrix(g::WeightedGraph)
    𝒮 = keys(g.states2ind)
    n = length(𝒮)
    T = zeros(n, n)
    for s in 𝒮
        for s′ in outneighbors(g, s)
            T[g.states2ind[s], g.states2ind[s′]] = get_weight(g, s, s′)
        end
    end
    return T
end

index(g::WeightedGraph, s) = g.states2ind[s]
state(g::WeightedGraph, i) = g.ind2states[i]