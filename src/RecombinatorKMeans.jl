module RecombinatorKMeans

using StatsBase
using SparseArrays
using Random
using Distributed

export kmeans, reckmeans, kmeans_randswap

Base.@propagate_inbounds function _cost(d1, d2)
    v1 = 0.0
    @simd for l = 1:length(d1)
        v1 += (d1[l] - d2[l])^2
    end
    return v1
end

let costsdict = Dict{Int,Vector{Float64}}(),
    countdict = Dict{Int,Vector{Int}}()
    global function get_costs!(c::Vector{Int}, data::Matrix{Float64}, centroids::Matrix{Float64}, inds = nothing)
        m, n = size(data)
        k = size(centroids, 2)
        @assert size(centroids, 1) == m
        inds = (inds ≡ nothing ? (1:n) : inds)
        nr = length(inds)

        costs = get!(costsdict, nr) do
            zeros(nr)
        end

        fill!(c, 1)
        i1 = 1
        @inbounds for i = inds
            v, x = Inf, 0
            for j = 1:k
                @views v1 = _cost(data[:,i], centroids[:,j])
                if v1 < v
                    v = v1
                    x = j
                end
            end
            costs[i1], c[i1] = v, x
            i1 += 1
        end

        return costs
    end

    global function assign_points!(c, data, centroids)
        k = size(centroids, 2)

        costs = get_costs!(c, data, centroids)
        return sum(costs)
    end

    global function recompute_centroids!(c, data, centroids)
        m, n = size(data)
        k = size(centroids, 2)
        @assert size(centroids, 1) == m
        @assert length(c) == n

        count = get!(countdict, k) do
            zeros(Int, k)
        end

        fill!(count, 0)
        fill!(centroids, 0.0)
        @inbounds for i = 1:n
            j = c[i]
            for l = 1:m
                centroids[l,j] += data[l,i]
            end
            count[j] += 1
        end
        @inbounds for j = 1:k
            z = count[j]
            z > 0 || continue
            for l = 1:m
                centroids[l,j] /= z
            end
        end

        return centroids
    end

    global function clear_cache!()
        empty!(costsdict)
        empty!(countdict)
    end
end

function init_centroid_unif(data::Matrix{Float64}, k::Int; w = nothing)
    m, n = size(data)
    centr = zeros(m, k)
    w::Vector{Float64} = w ≡ nothing ? ones(n) : w
    for j = 1:k
        i = sample(1:n, Weights(w))
        centr[:,j] .= data[:,i]
    end
    c = zeros(Int, size(data,2))
    costs = get_costs!(c, data, centr)
    return centr, c, costs, sum(costs)
end


function compute_costs_one!(costs::Vector{Float64}, data::Matrix{Float64}, x::AbstractVector{<:Float64})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m

    @inbounds for i = 1:n
        @views costs[i] = _cost(data[:,i], x)
    end
    return costs
end
compute_costs_one(data::Matrix{Float64}, x::AbstractVector{<:Float64}) = compute_costs_one!(Array{Float64}(undef,size(data,2)), data, x)

# This function was first written from scratch from the k-means++ paper, then improved after reading
# the corresponding scikit-learn's code at https://github.com/scikit-learn/scikit-learn, then heavily modified.
# The scikit-learn's version was first coded by Jan Schlueter as a port of some other code that is now lost.
function init_centroid_pp(pool::Matrix{Float64}, k::Int; ncandidates = nothing, w = nothing, data = nothing)
    m, np = size(pool)

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
        i_best = 0
        for i in candidates
            new_c .= c
            pooli = pool[:,i]
            new_costs .= costs
            @inbounds for i1 = 1:n
                @views v = _cost(data[:,i1], pooli)
                if v < new_costs[i1]
                    new_c[i1] = j
                    new_costs[i1] = v
                end
            end
            cost = sum(new_costs)
            if cost < cost_best
                cost_best = cost
                i_best = i
                new_costs_best, new_costs = new_costs, new_costs_best
                new_c_best, new_c = new_c, new_c_best
            end
        end
        @assert i_best ≠ 0 && cost_best < Inf
        centr[:,j] .= pool[:,i_best]
        costs, new_costs_best = new_costs_best, costs
        if dataispool
            pcosts = costs
        elseif j < k
            pcosts .= min.(pcosts, compute_costs_one(pool, @view(pool[:,i_best])))
        end
        c, new_c_best = new_c_best, c
    end
    return centr, c, costs, sum(costs)
end

"""
  kmeans(data::Matrix{Float64}, k::Integer; keywords...)

Runs k-means using Lloyd's algorithm on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column).
It returns: 1) a vector of labels (`n` integers from 1 to `k`); 2) a `d`×`k` matrix of centroids;
3) the final cost

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
"""
function kmeans(data::Matrix{Float64}, k::Integer;
                max_it::Integer = 1000,
                seed::Union{Integer,Nothing} = nothing,
                init::Union{String,Matrix{Float64}} = "++",
                tol::Float64 = 1e-5,
                verbose::Bool = true
               )
    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    if init isa String
        if init == "++"
            centroids, c, _, cost = init_centroid_pp(data, k)
        elseif init == "unif"
            centroids, c, _, cost = init_centroid_unif(data, k)
        else
            throw(ArgumentError("unrecognized init string \"$init\"; should be \"++\" or \"unif\""))
        end
    else
        size(init) == (m, k) || throw(ArgumentError("invalid size of init, expected $((m,k)), found: $(size(init))"))
        centroids = init
        c = zeros(Int, n)
        cost = assign_points!(c, data, centroids)
    end

    verbose && println("initial cost = $cost")

    converged = false
    it = 0
    for outer it = 1:max_it
        recompute_centroids!(c, data, centroids)
        new_cost = assign_points!(c, data, centroids)
        if new_cost ≥ cost * (1 - tol)
            cost = new_cost
            converged = true
            break
        end
        cost = new_cost
        verbose && println("it = $it cost = $cost")
    end
    verbose && println("final cost = $cost (" *
                       (converged ? "converged" : "did not converge") *
                       " in $it iterations)")
    return c, centroids, cost
end

struct ResultsRecKMeans
    exit_status::Symbol
    labels::Vector{Int}
    centroids::Matrix{Float64}
    cost::Float64
    all_costs::Union{Nothing,Vector{Vector{Float64}}}
end

"""
  reckmeans(data::Matrix{Float64}, k::Integer, Jlist; keywords...)

Runs recombinator-k-means on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column). The `Jlist`
paramter is either an integer or an iterable that specifies how many restarts to do for
each successive batch. If it is an integer, it just does the same number of batches indefinitely
until some stopping criterion gets triggered. If it is an iterable with a finite length,
e.g. `[10,5,2]`, the algorithm might exit after the end of the list instead.

It returns an object of type `ResultsRecKMeans`, which contains the following fields:
* exit_status: a symbol that indicates the reason why the algorithm stopped. It can take two
  values, `:collapsed` or `:maxiters`.
* labels: a vector of labels (`n` integers from 1 to `k`)
* centroids: a `d`×`k` matrix of centroids
* cost: the final cost
* all_costs: with the `keepallcosts==true` option, it holds a vector of vectors of all the costs
  found during the iteration (one vector per batch); otherwise `nothing`

The possible keyword arguments are:

* `Δβ`: a `Float64` (default=0.1), the reweigting parameter increment (only makes sense if
  non-negative).
* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `tol`: a `Float64` (default=1e-4), the relative tolerance for determining whether the solutions
  have collapsed.
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.
* `keepallcosts`: a `Bool` (default=`false`); if `true`, all the costs found during the iteration
  are kept and returned in the output.
* `max_it`: an `Int` (default=10)), maximum number of Lloyd (kmeans) iterations.
* `lltol`: a `Float64` (default=1e-5), relative tolerance for Lloyd (kmeans) convergence.

If there are workers available this function will parallelize each batch.
"""
function reckmeans(data::Matrix{Float64}, k::Integer, Jlist;
                   Δβ::Float64 = 0.1,
                   seed::Union{Integer,Nothing} = nothing,
                   tol::Float64 = 1e-4,
                   lltol::Float64 = 1e-5,
                   verbose::Bool = true,
                   keepallcosts::Bool = false,
                   max_it::Int = 10
                  )
    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    β = 0.0
    best_cost = Inf
    best_centr = Matrix{Float64}(undef, m, k)

    pool = data
    w = ones(size(pool, 2))
    allcosts = keepallcosts ? Vector{Float64}[] : nothing
    centroidsR = Matrix{Float64}[]
    costs = Float64[]
    exit_status = :running
    if Jlist isa Int
        Jlist = Iterators.repeated(Jlist)
    end
    for (it,J) in enumerate(Jlist)
        verbose && @info "it = $it J = $J"
        @assert length(w) == size(pool, 2)
        h0 = hash(pool)
        res = pmap(1:J) do a
            h = hash((seed, a), h0)
            Random.seed!(h)  # horrible hack to ensure determinism (not really required, only useful for experiments)
            centr, c, _, cost = init_centroid_pp(pool, k, w = w, data = data)
            for ll_it = 1:max_it
                recompute_centroids!(c, data, centr)
                new_cost = assign_points!(c, data, centr)
                if new_cost ≥ cost * (1 - lltol)
                    cost = new_cost
                    break
                end
                cost = new_cost
            end
            verbose && println("  a = $a cost = $cost")
            return centr, cost
        end
        append!(centroidsR, (r[1] for r in res))
        append!(costs, (r[2] for r in res))
        perm = sortperm(costs)
        centroidsR = centroidsR[perm[1:J]]
        costs = costs[perm[1:J]]

        keepallcosts && push!(allcosts, copy(costs))
        if costs[1] < best_cost
            best_cost = costs[1]
            best_centr .= centroidsR[1]
        end
        resize!(w, J*k)
        β += Δβ
        mean_cost = mean(costs)
        for a = 1:J
            w[(1:k) .+ (a-1)*k] .= exp(-β * ((costs[a] - best_cost) / (mean_cost - best_cost)))
        end
        w ./= sum(w)
        pool = hcat(centroidsR...)
        verbose && (@everywhere flush(stdout); println("  mean cost = $mean_cost best_cost = $best_cost"))
        if mean_cost ≤ best_cost * (1 + tol)
            verbose && @info "collapsed"
            exit_status = :collapsed
            break
        end
    end
    exit_status == :running && (exit_status = :maxiters)
    cC = zeros(Int, n)
    centroidsC = best_centr
    costC = assign_points!(cC, data, centroidsC)
    verbose && @info "final cost = $costC"

    clear_cache!()
    @sync for id in workers()
        @spawnat id clear_cache!()
    end

    return ResultsRecKMeans(
        exit_status,
        cC,
        centroidsC,
        costC,
        allcosts)
end

struct ResultsKMeansRS
    exit_status::Symbol
    labels::Vector{Int}
    centroids::Matrix{Float64}
    cost::Float64
    time::Float64
    iters::Int
    all_costs::Union{Nothing,Vector{Float64}}
    all_times::Union{Nothing,Vector{Float64}}
end


function kmeans_randswap(data::Matrix{Float64}, k::Integer;
                         max_it::Integer = 1000,
                         max_time::Float64 = Inf,
                         target_cost::Float64 = 0.0,
                         seed::Union{Integer,Nothing} = nothing,
                         ll_max_it::Integer = 2,
                         ll_tol::Float64 = 1e-5,
                         init::Union{String,Matrix{Float64}} = "++",
                         keepallcosts::Bool = false,
                         final_converge::Bool = true,
                         verbose::Bool = true
                        )
    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    t0 = time()

    if init isa String
        if init == "++"
            centroids, c, costs, cost = init_centroid_pp(data, k)
        elseif init == "unif"
            centroids, c, costs, cost = init_centroid_unif(data, k)
        else
            throw(ArgumentError("unrecognized init string \"$init\"; should be \"++\" or \"unif\""))
        end
    else
        size(init) == (m, k) || throw(ArgumentError("invalid size of init, expected $((m,k)), found: $(size(init))"))
        centroids = init
        c = zeros(Int, n)
        costs = get_costs!(c, data, centroids)
        cost = sum(costs)
    end

    verbose && @info "initial cost = $cost"
    allcosts = keepallcosts ? Float64[cost] : nothing
    alltimes = keepallcosts ? Float64[time() - t0] : nothing
    exit_status = :running
    new_centroids, new_c, new_costs = similar(centroids), similar(c), similar(costs)
    it = 0
    converged = false
    for outer it = 1:max_it
        new_centroids .= centroids
        new_c .= c
        new_costs .= costs
        j = rand(1:k)
        i = rand(1:n)
        xi = data[:,i]
        new_centroids[:,j] = xi

        msk = (c .== j)
        inds = findall(msk)
        c_msk = zeros(Int, length(inds))
        costs_msk = get_costs!(c_msk, data, new_centroids, inds)
        i_msk = 0
        @inbounds for i1 = 1:n
            if msk[i1]
                i_msk += 1
                new_costs[i1] = costs_msk[i_msk]
                new_c[i1] = c_msk[i_msk]
            else
                @views v = _cost(data[:,i1], xi)
                if v < new_costs[i1]
                    new_costs[i1] = v
                    new_c[i1] = j
                end
            end
        end
        new_cost = sum(new_costs)

        new_converged = false
        for ll_it = 1:ll_max_it
            recompute_centroids!(new_c, data, new_centroids)
            new_costs = get_costs!(new_c, data, new_centroids)
            new_cost2 = sum(new_costs)
            if new_cost2 ≥ new_cost * (1 - ll_tol)
                new_cost = new_cost2
                new_converged = true
                break
            end
            new_cost = new_cost2
        end
        if new_cost < cost
            cost = new_cost
            converged = new_converged
            centroids, new_centroids = new_centroids, centroids
            c, new_c = new_c, c
            costs, new_costs = new_costs, costs
            t = time() - t0
            verbose && println("it = $it cost = $cost time = $t")
            if keepallcosts
                append!(allcosts, cost)
                append!(alltimes, t)
            end
        end
        if cost ≤ target_cost
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
    exit_status == :running && (exit_status = :maxiters)
    if final_converge && !converged
        while true
            recompute_centroids!(c, data, centroids)
            new_cost = assign_points!(c, data, centroids)
            if new_cost ≥ cost * (1 - ll_tol)
                cost = new_cost
                break
            end
            cost = new_cost
        end
    end
    t = time() - t0
    verbose && @info "final cost = $cost time = $t"
    if keepallcosts
        append!(allcosts, cost)
        append!(alltimes, t)
    end
    return ResultsKMeansRS(
        exit_status,
        c,
        centroids,
        cost,
        t,
        it,
        allcosts,
        alltimes)
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
