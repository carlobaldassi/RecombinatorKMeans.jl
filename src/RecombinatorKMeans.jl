module RecombinatorKMeans

using Random
using Statistics
using StatsBase
using ExtractMacro

export kmeans, reckmeans, gakmeans, rswapkmeans

mutable struct Configuration
    m::Int
    k::Int
    n::Int
    c::Vector{Int}
    cost::Float64
    costs::Vector{Float64}
    centroids::Matrix{Float64}
    active::BitVector
    nonempty::BitVector
    csizes::Vector{Int}
    function Configuration(m::Int, k::Int, n::Int, c::Vector{Int}, costs::Vector{Float64}, centroids::Matrix{Float64})
        @assert length(c) == n
        @assert length(costs) == n
        @assert size(centroids) == (m, k)
        cost = sum(costs)
        active = trues(k)
        nonempty = trues(k)
        csizes = zeros(Int, k)
        if !all(c .== 0)
            for i = 1:n
                csizes[c[i]] += 1
            end
            nonempty .= csizes .> 0
        end
        return new(m, k, n, c, cost, costs, centroids, active, nonempty, csizes)
    end
    function Base.copy(config::Configuration)
        @extract config : m k n c cost costs centroids active nonempty csizes
        return new(m, k, n, copy(c), cost, copy(costs), copy(centroids), copy(active), copy(nonempty), copy(csizes))
    end
end

function Configuration(data::Matrix{Float64}, centroids::Matrix{Float64})
    m, n = size(data)
    k = size(centroids, 2)
    @assert size(centroids, 1) == m

    c = zeros(Int, n)
    costs = fill(Inf, n)
    config = Configuration(m, k, n, c, costs, centroids)
    partition_from_centroids!(config, data)
    return config
end

function Base.copy!(config::Configuration, other::Configuration)
    @extract config : m k n c costs centroids active nonempty csizes
    @extract other : om=m ok=k on=n oc=c ocosts=costs ocentroids=centroids oactive=active ononempty=nonempty ocsizes=csizes
    @assert m == om && k == ok && n == on
    copy!(c, oc)
    config.cost = other.cost
    copy!(costs, ocosts)
    copy!(centroids, ocentroids)
    copy!(active, oactive)
    copy!(nonempty, ononempty)
    copy!(csizes, ocsizes)
    return config
end


function remove_empty!(config::Configuration)
    @extract config: m k n c costs centroids active nonempty csizes

    k_new = sum(nonempty)
    if k_new == k
        return config
    end
    centroids = centroids[:, nonempty]
    new_inds = cumsum(nonempty)
    for i = 1:n
        @assert nonempty[c[i]]
        c[i] = new_inds[c[i]]
    end
    csizes = csizes[nonempty]
    nonempty = trues(k_new)

    config.k = k_new
    config.centroids = centroids
    config.csizes = csizes
    config.nonempty = nonempty

    return config
end

Base.@propagate_inbounds function _cost(d1, d2)
    v1 = 0.0
    @simd for l = 1:length(d1)
        v1 += (d1[l] - d2[l])^2
    end
    return v1
end

# grouping method from Kaukoranta, Fränti, Nevlainen, "Reduced comparison for the exact GLA"
function partition_from_centroids!(config::Configuration, data::Matrix{Float64})
    @extract config: m k n c costs centroids active nonempty csizes
    @assert size(data) == (m, n)

    active_inds = findall(active)
    all_inds = collect(1:k)

    fill!(nonempty, false)
    fill!(csizes, 0)

    num_fullsearch = 0
    cost = 0.0
    @inbounds for i in 1:n
        ci = c[i]
        if ci > 0 && active[ci]
            old_v′ = costs[i]
            @views new_v′ = _cost(data[:,i], centroids[:,ci])
            fullsearch = (new_v′ > old_v′)
        else
            fullsearch = (ci == 0)
        end
        num_fullsearch += fullsearch

        v, x, inds = fullsearch ? (Inf, 0, all_inds) : (costs[i], ci, active_inds)
        for j in inds
            @views v′ = _cost(data[:,i], centroids[:,j])
            if v′ < v
                v, x = v′, j
            end
        end
        costs[i], c[i] = v, x
        nonempty[x] = true
        csizes[x] += 1
        cost += v
    end

    config.cost = cost
    return config
end


let centroidsdict = Dict{NTuple{2,Int},Matrix{Float64}}()

    global function centroids_from_partition!(config::Configuration, data::Matrix{Float64})
        @extract config: m k n c costs centroids active nonempty csizes
        @assert size(data) == (m, n)

        new_centroids = get!(centroidsdict, (m,k)) do
            zeros(Float64, m, k)
        end
        fill!(new_centroids, 0.0)
        fill!(active, false)
        @inbounds for i = 1:n
            j = c[i]
            for l = 1:m
                new_centroids[l,j] += data[l,i]
            end
        end
        @inbounds for j = 1:k
            z = csizes[j]
            z > 0 || continue
            @assert nonempty[j]
            for l = 1:m
                new_centroids[l,j] /= z
                new_centroids[l,j] ≠ centroids[l,j] && (active[j] = true)
                centroids[l,j] = new_centroids[l,j]
            end
        end
        return config
    end

    global function clear_cache!()
        empty!(centroidsdict)
    end
end

function check_empty!(config::Configuration, data::Matrix{Float64})
    @extract config: m k n c costs centroids active nonempty csizes
    num_nonempty = sum(nonempty)
    num_centroids = min(config.n, config.k)
    gap = num_centroids - num_nonempty
    gap == 0 && return false
    to_fill = findall(.~(nonempty))[1:gap]
    for j in to_fill
        i = rand(1:n)
        centroids[:,j] .= data[:,i]
        active[j] = true
    end
    return true
end



function init_centroid_unif(data::Matrix{Float64}, k::Int; kw...)
    m, n = size(data)
    centroids = zeros(m, k)
    for j = 1:k
        i = rand(1:n)
        centroids[:,j] .= data[:,i]
    end
    return Configuration(data, centroids)
end

function compute_costs_one!(costs::Vector{Float64}, data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m

    @inbounds for i = 1:n
        @views costs[i] = _cost(data[:,i], x)
    end
    return costs
end
compute_costs_one(data::AbstractMatrix{<:Float64}, x::AbstractVector{<:Float64}) = compute_costs_one!(Array{Float64}(undef,size(data,2)), data, x)

# This function was first written from scratch from the k-means++ paper, then improved after reading
# the corresponding scikit-learn's code at https://github.com/scikit-learn/scikit-learn, then heavily modified.
# The scikit-learn's version was first coded by Jan Schlueter as a port of some other code that is now lost.
function init_centroid_pp(pool::Matrix{Float64}, k::Int; ncandidates = nothing, w = nothing, data = nothing)
    m, np = size(pool)
    @assert np ≥ k

    ncandidates::Int = ncandidates ≡ nothing ? floor(Int, 2 + log(k)) : ncandidates
    w = (w ≡ nothing ? ones(np) : w)::Vector{Float64}
    data = (data ≡ nothing ? pool : data)::Matrix{Float64}
    dataispool = data ≡ pool
    n = size(data, 2)
    @assert size(data, 1) == m

    centr = zeros(m, k)
    i = sample(1:np, Weights(w))
    pooli = pool[:,i]
    centr[:,1] .= pooli

    costs = compute_costs_one(data, pooli)
    pcosts = dataispool ? costs : compute_costs_one(pool, pooli)

    curr_cost = sum(costs)
    c = ones(Int, n)

    new_costs, new_c = similar(costs), similar(c)
    new_costs_best, new_c_best = similar(costs), similar(c)
    for j = 2:k
        pw = Weights(pcosts .* w)
        nonz = count(pw .≠ 0)
        candidates = sample(1:np, pw, min(ncandidates,np,nonz), replace = false)
        cost_best = Inf
        y_best = 0
        for y in candidates
            pooly = pool[:,y]
            compute_costs_one!(new_costs, data, pooly)
            cost = 0.0
            @inbounds for i = 1:n
                v = new_costs[i]
                v′ = costs[i]
                if v < v′
                    new_c[i] = j
                    cost += v
                else
                    new_costs[i] = v′
                    new_c[i] = c[i]
                    cost += v′
                end
            end
            if cost < cost_best
                cost_best = cost
                y_best = y
                new_costs_best, new_costs = new_costs, new_costs_best
                new_c_best, new_c = new_c, new_c_best
            end
        end
        @assert y_best ≠ 0 && cost_best < Inf
        pooly = pool[:,y_best]
        centr[:,j] .= pooly
        costs, new_costs_best = new_costs_best, costs
        if dataispool
            pcosts = costs
        elseif j < k
            pcosts .= min.(pcosts, compute_costs_one(pool, pooly))
        end
        c, new_c_best = new_c_best, c
        dataispool && (pc = c)
    end
    return Configuration(m, k, n, c, costs, centr)
end

"""
  kmeans(data::Matrix{Float64}, k::Integer; keywords...)

Runs k-means using Lloyd's algorithm on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column).
It returns: 1) a vector of labels (`n` integers from 1 to `k`); 2) a `d`×`k` matrix of centroids;
3) the final cost; 4) whether it converged or not

The possible keyword arguments are:

* `max_it`: maximum number of iterations (default=1000). Normally the algorithm stops at fixed
  points.
* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `init`: how to initialize (default=`"++"`). It can be a string or a `Matrix{Float64}`. If it's
  a matrix, it represents the initial centroids (by column). If it is a string, it can be either
  `"++"` for k-means++ initialization, or `"unif"` for uniform.
* `tol`: a `Float64`, relative tolerance for detecting convergence (default=1e-5).
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.
* `ncandidates`: if init=="++", set the number of candidates for k-means++ (the default is
  `nothing`, which means that it is set automatically)
"""
function kmeans(
        data::Matrix{Float64}, k::Integer;
        max_it::Integer = 1000,
        seed::Union{Integer,Nothing} = nothing,
        init::Union{String,Matrix{Float64}} = "++",
        tol::Float64 = 1e-5,
        verbose::Bool = true,
        ncandidates::Union{Nothing,Int} = nothing,
    )
    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    if init isa String
        if init == "++"
            config = init_centroid_pp(data, k; ncandidates)
        elseif init == "unif"
            config = init_centroid_unif(data, k)
        else
            throw(ArgumentError("unrecognized init string \"$init\"; should be \"++\" or \"unif\""))
        end
    else
        size(init) == (m, k) || throw(ArgumentError("invalid size of init, expected $((m,k)), found: $(size(init))"))
        centroids = copy(init)
        config = Configuration(data, centroids)
    end

    verbose && println("initial cost = $(config.cost)")
    converged = lloyd!(config, data, max_it, tol, verbose)

    return config.c, config.centroids, config.cost, converged
end



function crossover_GA(configs::Vector{Configuration}, data::Matrix{Float64}, k::Int, a::Int; kw...)
    Jc = length(configs)
    (i1, i2) = [(x,y) for x = 1:Jc for y = (x+1):Jc][a] # TODO: quite ugly...
    return merge_pnn(configs[i1], configs[i2], data, k)
end


function lloyd!(config::Configuration, data::Matrix{Float64}, max_it::Int, tol::Float64, verbose::Bool)
    cost0 = config.cost
    converged = false
    it = 0
    for outer it = 1:max_it
        centroids_from_partition!(config, data)
        old_cost = config.cost
        found_empty = check_empty!(config, data)
        partition_from_centroids!(config, data)
        new_cost = config.cost
        if new_cost ≥ old_cost * (1 - tol) && !found_empty
            verbose && println("converged cost = $new_cost")
            converged = true
            break
        end
        verbose && println("lloyd it = $it cost = $new_cost")
    end
    return converged
end

function combine_configs(config1::Configuration, config2::Configuration)
    @extract config1 : m1=m k1=k n1=n c1=c costs1=costs centroids1=centroids
    @extract config2 : m2=m k2=k n2=n c2=c costs2=costs centroids2=centroids
    @assert (m1, n1) == (m2, n2)
    m, n = m1, n1
    k = k1 + k2
    centroids_new = hcat(centroids1, centroids2)
    @assert size(centroids_new) == (m, k)
    c_new = zeros(Int, n)
    costs_new = zeros(n)
    for i = 1:n
        j1, j2 = c1[i], c2[i]
        cost1, cost2 = costs1[i], costs2[i]
        if cost1 ≤ cost2
            j = j1
            cost = cost1
        else
            j = k1 + j2
            cost = cost2
        end
        c_new[i], costs_new[i] = j, cost
    end
    return Configuration(m, k, n, c_new, costs_new, centroids_new)
end

function pairwise_nn!(config::Configuration, tgt_k::Int)
    @extract config : m k n c costs centroids csizes
    if k < tgt_k
        @assert false # TODO: inflate the config? this shouldn't normally happen
    end
    if k == tgt_k
        return config
    end

    nns = zeros(Int, k)
    nns_costs = fill(Inf, k)
    @inbounds for j = 1:k
        z = csizes[j]
        v, x = Inf, 0
        for j′ = 1:k
            j′ == j && continue
            z′ = csizes[j′]
            @views v1 = (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
            if v1 < v
                v = v1
                x = j′
            end
        end
        nns_costs[j], nns[j] = v, x
    end

    @inbounds while k > tgt_k
        jm = findmin(@view(nns_costs[1:k]))[2]
        js = nns[jm]
        @assert nns_costs[js] == nns_costs[jm]
        @assert jm < js

        # update centroid
        zm, zs = csizes[jm], csizes[js]
        for l = 1:m
            centroids[l,jm] = (zm * centroids[l,jm] + zs * centroids[l,js]) / (zm + zs)
            centroids[l,js] = centroids[l,k]
        end

        # update csizes
        csizes[jm] += zs
        csizes[js] = csizes[k]
        csizes[k] = 0

        # update partition
        # not needed since we don't use c in this computation and
        # we call partition_from_centroids! right after this function
        # for i = 1:n
        #     ci = c[i]
        #     if ci == js
        #         c[i] = jm
        #     elseif ci == k
        #         c[i] = js
        #     end
        # end

        # update nns
        nns[js] = nns[k]
        nns[k] = 0
        nns_costs[js] = nns_costs[k]
        nns_costs[k] = Inf

        num_fullupdates = 0
        for j = 1:(k-1)
            # 1) merged cluster jm, or clusters which used to point to either jm or js
            #    perform a full update
            if j == jm || nns[j] == jm || nns[j] == js
                num_fullupdates += 1
                z = csizes[j]
                v, x = Inf, 0
                for j′ = 1:(k-1)
                    j′ == j && continue
                    z′ = csizes[j′]
                    @views v1 = (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
                    if v1 < v
                        v, x = v1, j′
                    end
                end
                nns_costs[j], nns[j] = v, x
            # 2) clusters that did not point to jm or js
            #    only compare the old cost with the cost for the updated cluster
            else
                z = csizes[j]
                # note: what used to point at k now must point at js
                v, x = nns_costs[j], (nns[j] ≠ k ? nns[j] : js)
                j′ = jm
                z′ = csizes[j′]
                @views v′ = (z * z′) / (z + z′) * _cost(centroids[:,j], centroids[:,j′])
                if v′ < v
                    v, x = v′, j′
                end
                nns_costs[j], nns[j] = v, x
                @assert nns[j] ≠ j
            end
        end

        k -= 1
    end

    config.k = k
    config.centroids = centroids[:,1:k]
    config.csizes = csizes[1:k]
    @assert all(config.csizes .> 0)
    config.active = trues(k)
    config.nonempty = trues(k)
    fill!(config.c, 0) # reset in order for partition_from_centroids! to work

end

function merge_pnn(config1::Configuration, config2::Configuration, data::Matrix{Float64}, tgt_k::Int)
    config = combine_configs(config1, config2)
    centroids_from_partition!(config, data)
    remove_empty!(config)
    pairwise_nn!(config, tgt_k)
    partition_from_centroids!(config, data)
    return config
end

struct Results
    exit_status::Symbol
    labels::Vector{Int}
    centroids::Matrix{Float64}
    cost::Float64
end

"""
  reckmeans(data::Matrix{Float64}, k::Integer, J::Integer; keywords...)

Runs recombinator-k-means on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column). `k` is
the number of clusters, `J` is the population size

It returns an object of type `Results`, which contains the following fields:
* exit_status: a symbol that indicates the reason why the algorithm stopped. It can take three
  values, `:collapsed`, `:maxgenerations` or `:noimprovement`.
* labels: a vector of labels (`n` integers from 1 to `k`)
* centroids: a `d`×`k` matrix of centroids
* cost: the final cost

The possible keyword arguments are:

* `Δβ`: a `Float64` (default=0.1), the reweigting parameter increment (only makes sense if
  non-negative).
* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `tol`: a `Float64` (default=1e-4), the relative tolerance for determining whether the solutions
  have collapsed.
* `lltol`: a `Float64` (default=1e-5), relative tolerance for Lloyd (kmeans) convergence.
* `max_it`: an `Int` (default=10), maximum number of Lloyd (kmeans) iterations.
* `max_gen`: an `Int` (default=1_000), maximum number of generations.
* `stop_if_noimprovement`: a `Bool` (default=false), stop if the new generation did not improve on
  the old one
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.

If there are multiple threads available this function will parallelize each batch.
"""
function reckmeans(
        data::Matrix{Float64}, k::Integer, J::Int;
        Δβ::Float64 = 0.1,
        seed::Union{Integer,Nothing} = nothing,
        tol::Float64 = 1e-4,
        lltol::Float64 = 1e-5,
        verbose::Bool = true,
        max_it::Int = 10,
        stop_if_noimprovement::Bool = false,
        max_gen::Int = 1_000
    )
    J ≥ 2 || throw(ArgumentError("J must be at least 2"))

    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    β = 0.0
    best_cost = Inf
    local best_config::Configuration

    pool = data
    w = ones(size(pool, 2))

    configs = Configuration[]

    exit_status = :running
    for generation = 0:max_gen
        verbose && @info "gen = $generation"
        @assert length(w) == size(pool, 2)
        h0 = hash(pool)
        new_configs = Vector{Configuration}(undef, J)
        Threads.@threads for a = 1:J
            h = hash((seed, a), h0)
            Random.seed!(h)  # horrible hack to ensure determinism (not really required, only useful for experiments)
            config = init_centroid_pp(pool, k; w, data)

            converged = lloyd!(config, data, max_it, lltol, false)
            verbose && println("  a = $a cost = $(config.cost)")
            new_configs[a] = config
        end
        append!(configs, new_configs)
        sort!(configs, by=config->config.cost)
        configs = configs[1:J]

        if configs[1].cost < best_cost
            best_cost = configs[1].cost
            best_config = copy(configs[1])
            improved = true
        else
            improved = false
        end
        mean_cost = mean(config.cost for config in configs)
        stddev_cost = std((config.cost for config in configs), mean = mean_cost)
        verbose && println("  best_cost = $best_cost mean cost = $mean_cost ± $stddev_cost")
        if stop_if_noimprovement && !improved
            verbose && @info "no improvement"
            exit_status = :noimprovement
            break
        end
        if mean_cost ≤ best_cost * (1 + tol)
            verbose && @info "collapsed"
            exit_status = :collapsed
            break
        end

        if generation < max_gen
            pool = hcat((config.centroids for config in configs)...)
            resize!(w, size(pool,2))
            β += Δβ
            for a = 1:(length(w) ÷ k)
                w[(1:k) .+ (a-1)*k] .= exp(-β * ((configs[a].cost - best_cost) / (mean_cost - best_cost)))
            end
            # w ./= sum(w)
        end
    end
    exit_status == :running && (exit_status = :maxgenerations)

    verbose && @info "final cost = $best_cost"

    clear_cache!()

    return Results(
        exit_status,
        best_config.c,
        best_config.centroids,
        best_config.cost,
        )
end

# find c such that: (c * (c-1)) ÷ 2 ≥ J
elitist_crossset_size(J) = ceil(Int, (1 + √(1 + 8J)) / 2)


"""
  gakmeans(data::Matrix{Float64}, k::Integer, J::Integer; keywords...)

Runs ga-k-means on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column). `k` is
the number of clusters, `J` is the population size

It returns an object of type `Results`, which contains the following fields:
* exit_status: a symbol that indicates the reason why the algorithm stopped. It can take three
  values, `:collapsed`, `:maxgenerations` or `:noimprovement`.
* labels: a vector of labels (`n` integers from 1 to `k`)
* centroids: a `d`×`k` matrix of centroids
* cost: the final cost

The possible keyword arguments are:

* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `init`: a `String` (default=`"++"`). It can be either `"++"` for k-means++ initialization,
  or `"unif"` for uniform+Lloyd, `"raw"` for uniform (non-optimized).
* `tol`: a `Float64` (default=1e-4), the relative tolerance for determining whether the solutions
  have collapsed.
* `lltol`: a `Float64` (default=1e-5), relative tolerance for Lloyd (kmeans) convergence.
* `max_it`: an `Int` (default=10), maximum number of Lloyd (kmeans) iterations.
* `max_gen`: an `Int` (default=1_000), maximum number of generations.
* `stop_if_noimprovement`: a `Bool` (default=true), stop if the new generation did not improve on
  the old one
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.

If there are multiple threads available this function will parallelize each batch.
"""
function gakmeans(
        data::Matrix{Float64}, k::Integer, J::Int;
        seed::Union{Integer,Nothing} = nothing,
        init::String = "++",
        tol::Float64 = 1e-4,
        lltol::Float64 = 1e-5,
        verbose::Bool = true,
        max_it::Integer = 10,
        max_gen::Integer = 1_000,
        stop_if_noimprovement::Bool = true,
    )
    init ∈ ["++", "unif", "raw"] || throw(ArgumentError("init must be one of \"++\", \"unif\" or \"raw\""))
    J ≥ 2 || throw(ArgumentError("J must be at least 2"))

    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    Jc = elitist_crossset_size(J)

    init_func = init == "++" ? init_centroid_pp :
                init_centroid_unif

    configs = Vector{Configuration}(undef, J)

    h0 = hash(data)

    exit_status = :running
    Threads.@threads for a = 1:J
        h = hash((seed, a), h0)
        Random.seed!(h)  # horrible hack to ensure determinism (not really required, only useful for experiments)
        config = init_func(data, k)
        init ≠ "raw" && (converged = lloyd!(config, data, max_it, lltol, false))
        configs[a] = config
    end
    sort!(configs, by=config->config.cost)
    costs = [config.cost for config in configs]
    verbose && @info "costs after init: $(costs)"

    best_cost = configs[1].cost
    best_config = copy(configs[1])

    mean_cost = mean(config.cost for config in configs)
    stddev_cost = std((config.cost for config in configs), mean = mean_cost)

    for generation = 1:max_gen
        old_costs = costs
        elite_configs = configs[1:Jc]
        new_configs = Vector{Configuration}(undef, J)
        h0 = hash(configs)
        Threads.@threads for a = 1:J
            h = hash((seed, a), h0)
            Random.seed!(h)  # horrible hack to ensure determinism (not really required, only useful for experiments)
            config::Configuration = crossover_GA(elite_configs, data, k, a)
            converged = lloyd!(config, data, max_it, lltol, false)

            verbose && println("  a = $a cost = $(config.cost)")
            new_configs[a] = config
        end
        sort!(new_configs, by=config->config.cost) # TODO ? the scaling is bad, but in practice this is negligible
        configs = new_configs
        costs = [config.cost for config in configs]

        if configs[1].cost < best_cost
            best_cost = configs[1].cost
            best_config = copy(configs[1])
            improved = true
        else
            improved = false
        end

        verbose && @info "costs after gen $generation: $costs"
        mean_cost = mean(costs)
        stddev_cost = std(costs, mean = mean_cost)
        verbose && println("  best_cost = $best_cost mean cost = $mean_cost ± $stddev_cost")

        ## same condition as in the original implementation
        if stop_if_noimprovement && !improved && generation > 2
            verbose && @info "no improvement"
            exit_status = :noimprovement
            break
        end
        if mean_cost ≤ best_cost * (1 + tol)
            verbose && @info "collapsed"
            exit_status = :collapsed
            break
        end
    end
    if exit_status == :running
        exit_status = :maxgenerations
    end

    verbose && @info "final cost = $best_cost"

    clear_cache!()

    return Results(
        exit_status,
        best_config.c,
        best_config.centroids,
        best_config.cost,
        )
end


"""
  rswapkmeans(data::Matrix{Float64}, k::Integer; keywords...)

Runs random-swap-k-means on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column). `k` is
the number of clusters, `J` is the population size

It returns an object of type `Results`, which contains the following fields:
* exit_status: a symbol that indicates the reason why the algorithm stopped. It can take three
  values, `:solved`, `:maxswaps` or `:outoftime`.
* labels: a vector of labels (`n` integers from 1 to `k`)
* centroids: a `d`×`k` matrix of centroids
* cost: the final cost

The possible keyword arguments are:

* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `init`: a `String` (default=`"++"`). It can be either `"++"` for k-means++ initialization,
  or `"unif"` for uniform+Lloyd.
* `lltol`: a `Float64` (default=1e-5), relative tolerance for Lloyd (kmeans) convergence.
* `max_it`: an `Int` (default=2), maximum number of Lloyd (kmeans) iterations.
* `max_swaps`: an `Int` (default=1_000), maximum number of swaps to attempt.
* `max_time`: a `Float64` (default=Inf), maximum allowed (wall-clock) time.
* `target_cost`: a `Float64` (default=0.0), stop if the cost achieves this value (the problem is
  considered solved)
* `final_convergence`: a `Bool`; if `true` (the default) a final run of Lloyd (kmeans) with
  unbounded iterations is performed until convergence, in case it hadn't already been achieved
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.
"""
function rswapkmeans(
        data::Matrix{Float64}, k::Integer;
        seed::Union{Integer,Nothing} = nothing,
        init::String = "++",
        lltol::Float64 = 1e-5,
        max_swaps::Integer = 1_000,
        max_time::Float64 = Inf,
        target_cost::Float64 = 0.0,
        max_it::Integer = 2,
        final_converge::Bool = true,
        verbose::Bool = true,
    )
    init ∈ ["++", "unif"] || throw(ArgumentError("init must be one of \"++\" or \"unif\""))

    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    t0 = time()

    init_func = init == "++" ? init_centroid_pp :
                init_centroid_unif

    config = init_func(data, k)
    converged = lloyd!(config, data, max_swaps, lltol, false)

    verbose && @info "cost after init: $(config.cost)"

    exit_status = :running
    best_config = copy(config)
    best_converged = converged
    it = 0
    for outer it = 1:max_swaps
        copy!(config, best_config)

        j = rand(1:k)
        i = rand(1:n)
        config.centroids[:,j] = @view data[:,i]
        config.active[j] = true
        partition_from_centroids!(config, data)

        converged = lloyd!(config, data, max_it, lltol, false)

        if config.cost < best_config.cost
            best_config, config = config, best_config
            best_converged = converged
            t = time() - t0
            verbose && println("  it = $it cost = $(best_config.cost) time = $t")
        end
        if best_config.cost ≤ target_cost
            exit_status = :solved
            verbose && @info "target_cost achieved"
            break
        end
        if time() - t0 ≥ max_time
            exit_status = :outoftime
            verbose && @info "max_time reached"
            break
        end
    end
    if exit_status == :running
        exit_status = :maxswaps
    end
    if final_converge && !best_converged
        best_converged = lloyd!(best_config, data, typemax(Int), lltol, false)
    end
    t = time() - t0
    verbose && @info "final cost = $(best_config.cost) time = $t"

    clear_cache!()

    return Results(
        exit_status,
        best_config.c,
        best_config.centroids,
        best_config.cost
      )
end


# Centroid Index
# P. Fränti, M. Rezaei and Q. Zhao, Centroid index: cluster level similarity measure, Pattern Recognition, 2014
function CI(true_centroids::Matrix{Float64}, centroids::Matrix{Float64})
    m, tk = size(true_centroids)
    @assert size(centroids, 1) == m
    k = size(centroids, 2)

    matched = falses(tk)
    @inbounds for j = 1:k
        v = Inf
        p = 0
        for tj = 1:tk
            @views v1 = _cost(true_centroids[:,tj], centroids[:,j])
            if v1 < v
                v = v1
                p = tj
            end
        end
        matched[p] = true
    end

    return tk - count(matched)
end

# CI_sym(centroids1::Matrix{Float64}, centroids2::Matrix{Float64}) =
#     max(CI(centroids1, centroids2), CI(centroids2, centroids1))

function CI_sym(centroids1::Matrix{Float64}, centroids2::Matrix{Float64})
    m, k1 = size(centroids1)
    @assert size(centroids2, 1) == m
    k2 = size(centroids2, 2)

    a12 = zeros(Int, k1)
    a21 = zeros(Int, k2)
    v12 = fill(Inf, k1)
    v21 = fill(Inf, k2)
    @inbounds for j1 = 1:k1, j2 = 1:k2
        @views v = _cost(centroids1[:,j1], centroids2[:,j2])
        if v < v12[j1]
            v12[j1] = v
            a12[j1] = j2
        end
        if v < v21[j2]
            v21[j2] = v
            a21[j2] = j1
        end
    end

    return max(k1 - length(BitSet(a12)), k2 - length(BitSet(a21)))
end

# Variation of Information
# M. Meilă, Comparing clusterings—an information based distance, Journal of multivariate analysis, 2007

xlogx(x) = ifelse(iszero(x), zero(x), x * log(x))
entropy(p) = -sum(xlogx, p)

function VI(c1::Vector{Int}, c2::Vector{Int})
    n = length(c1)
    length(c2) == n || throw(ArgumentError("partitions bust have the same length, given: $(n) and $(length(c2))"))
    a, k1 = extrema(c1)
    a ≥ 1 || throw(ArgumentError("partitions elements must be ≥ 1, found $(a)"))
    a, k2 = extrema(c2)
    a ≥ 1 || throw(ArgumentError("partitions elements must be ≥ 1, found $(a)"))
    o = zeros(k1, k2)
    o1 = zeros(k1)
    o2 = zeros(k2)
    for i = 1:n
        j1, j2 = c1[i], c2[i]
        o[j1, j2] += 1
        o1[j1] += 1
        o2[j2] += 1
    end
    o ./= n
    o1 ./= n
    o2 ./= n
    vi = 2entropy(o) - entropy(o1) - entropy(o2)
    @assert vi > -1e-12
    return max(0.0, vi)
end

end # module
