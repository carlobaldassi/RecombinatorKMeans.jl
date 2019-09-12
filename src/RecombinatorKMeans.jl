module RecombinatorKMeans

using StatsBase
using SparseArrays
using Random
using Distributed

export kmeans, reckmeans

let costsdict = Dict{Int,Matrix{Float64}}(),
    countdict = Dict{Int,Vector{Float64}}()
    global function get_costs!(c, data, centroids)
        m, n = size(data)
        k = size(centroids, 2)
        @assert size(centroids, 1) == m

        costs = get!(costsdict, n) do
            zeros(1, n)
        end

        fill!(c, 1)
        @inbounds for i = 1:n
            v = Inf
            x = 0
            for j = 1:k
                v1 = 0.0
                @simd for l = 1:m
                    v1 += (data[l,i] - centroids[l,j])^2
                end
                if v1 < v
                    v = v1
                    x = j
                end
            end
            costs[i] = v
            c[i] = x
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
    return centr
end


function compute_costs_one!(costs::Vector{Float64}, data::Matrix{Float64}, x::Vector{Float64})
    m, n = size(data)
    @assert length(costs) == n
    @assert length(x) == m

    @inbounds for i = 1:n
        v = 0.0
        @simd for l = 1:m
            v += (data[l,i] - x[l])^2
        end
        costs[i] = v
    end
    return costs
end
compute_costs_one(data::Matrix{Float64}, x::Vector{Float64}) = compute_costs_one!(Array{Float64}(undef,size(data,2)), data, x)

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
    new_pcosts_best = dataispool ? new_costs_best : similar(pcosts)
    for j = 2:k
        pw = Weights(pcosts .* w)
        candidates = [sample(1:np, pw) for _ = 1:ncandidates]
        cost_best = Inf
        i_best = 0
        for i in candidates
            new_c .= c
            pooli = pool[:,i]
            new_costs .= costs
            @inbounds for i1 = 1:n
                v = 0.0
                @simd for l = 1:m
                    v += (data[l,i1] - pooli[l])^2
                end
                if v < costs[i1]
                    new_c[i1] = j
                    new_costs[i1] = v
                end
            end
            cost = sum(new_costs)
            if cost < cost_best
                cost_best = cost
                i_best = i
                new_costs_best .= new_costs
                if !dataispool
                    new_pcosts_best .= min.(pcosts, compute_costs_one(pool, pooli))
                end
                new_c_best .= new_c
            end
        end
        @assert i_best ≠ 0 && cost_best < Inf
        centr[:,j] .= pool[:,i_best]
        costs .= new_costs_best
        if !dataispool
            pcosts .= new_pcosts_best
        end
        c .= new_c_best
    end
    return centr
end

"""
  kmeans(data::Matrix{Float64}, k::Integer; keywords...)

Runs k-means using Lloyd's algorithm on the given data matrix (if the size is `d`×`n` then `d` is
the dimension and `n` the number of points, i.e. data is organized by column).
It returns: 1) a vector of labels (`n` integers from 1 to `k`); 2) a `d`×`k` matrix of centroids;
3) the final loss

The possible keyword arguments are:

* `max_it`: maximum number of iterations (default=1000). Normally the algorithm stops at fixed
  points.
* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `init`: how to initialize (default=`"++"`). It can be a string or a `Matrix{Float64}`. If it's
  a matrix, it represents the initial centroids (by column). If it is a string, it can be either
  `"++"` for k-means++ initialization, or `"unif"` for uniform.
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.
"""
function kmeans(data::Matrix{Float64}, k::Integer;
                max_it::Integer = 1000,
                seed::Union{Integer,Nothing} = nothing,
                init::Union{String,Matrix{Float64}} = "++",
                verbose::Bool = true
               )
    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    if init isa String
        if init == "++"
            centroids = init_centroid_pp(data, k)
        elseif init == "unif"
            centroids = init_centroid_unif(data, k)
        else
            throw(ArgumentError("unrecognized init string \"$init\"; should be \"++\" or \"unif\""))
        end
    else
        size(init) == (m, k) || throw(ArgumentError("invalid size of init, expected $((m,k)), found: $(size(init))"))
        centroids = init
    end

    c = zeros(Int, n)

    cost = assign_points!(c, data, centroids)
    verbose && println("initial cost = $cost")

    for it = 1:max_it
        recompute_centroids!(c, data, centroids)
        new_cost = assign_points!(c, data, centroids)
        new_cost == cost && break
        cost = new_cost
        verbose && println("it = $it cost = $cost")
    end
    return c, centroids, cost
end

struct ResultsRecKMeans
    exit_status::Symbol
    labels::Vector{Int}
    centroids::Matrix{Float64}
    loss::Float64
    all_losses::Union{Nothing,Vector{Vector{Float64}}}
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
* exit_status: a symbol that indicates the reason why the algorithm stopped. It can take three
  values, `:collapsed`, `:didntimprove` or `:maxiters`.
* labels: a vector of labels (`n` integers from 1 to `k`)
* centroids: a `d`×`k` matrix of centroids
* loss: the final loss
* all_losses: either `nothing`, or a vector of vectors of all the losses found during the
  iteration (one vector per batch), depending on the value of the `keepalllosses` option.

The possible keyword arguments are:

* `β`: a `Float64` (default=7.5), the reweigting parameter (only makes sense if non-negative)
* `seed`: random seed, either an integer or `nothing` (this is the default, it means no seeding
  is performed).
* `tol`: a `Float64` (default=1e-4), the relative tolerance for determining whether the solutions
  have collapsed or have stopped improving
* `verbose`: a `Bool`; if `true` (the default) it prints information on screen.
* `keepalllosses`: a `Bool` (default=`false`); if `true`, the returned structure contains all the
  losses found during the iteration.

If there are workers available this function will parallelize each batch.
"""
function reckmeans(data::Matrix{Float64}, k::Integer, Jlist;
                   β = 7.5,
                   seed::Union{Integer,Nothing} = nothing,
                   tol::Float64 = 1e-4,
                   verbose::Bool = true,
                   keepalllosses::Bool = false
                  )
    seed ≢ nothing && Random.seed!(seed)
    m, n = size(data)

    best_cost = Inf
    best_centr = Matrix{Float64}(undef, m, k)

    dd = copy(data)
    w = ones(size(dd, 2))
    old_mean_cost = Inf
    old_best_cost = Inf
    allcosts = keepalllosses ? Vector{Float64}[] : nothing
    exit_status = :running
    if Jlist isa Int
        Jlist = Iterators.repeated(Jlist)
    end
    for (it,J) in enumerate(Jlist)
        verbose && @info "it = $it J = $J"
        @assert length(w) == size(dd, 2)
        h0 = hash(dd)
        res = pmap(1:J) do a
            h = hash((seed, a), h0)
            Random.seed!(h)  # horrible hack to ensure determinism (not really required, only useful for experiments)
            centr = init_centroid_pp(dd, k, w = w, data = data)
            c = zeros(Int, n)
            cost = assign_points!(c, data, centr)
            while true
                recompute_centroids!(c, data, centr)
                new_cost = assign_points!(c, data, centr)
                new_cost == cost && break
                cost = new_cost
            end
            verbose && println("  a = $a cost = $cost")
            return centr, cost
        end
        centroidsR = [r[1] for r in res]
        costs = [r[2] for r in res]
        keepalllosses && push!(allcosts, costs)
        batch_best_cost, a_opt = findmin(costs)
        if batch_best_cost < best_cost
            best_cost = batch_best_cost
            best_centr .= centroidsR[a_opt]
        end
        resize!(w, J*k)
        mean_cost = mean(costs)
        for a = 1:J
            w[(1:k) .+ (a-1)*k] .= exp(-β * ((costs[a] - batch_best_cost) / (mean_cost - batch_best_cost)))
        end
        w ./= sum(w)
        dd = hcat(centroidsR...)
        verbose && (@everywhere flush(stdout); println("  mean cost = $mean_cost best_cost = $best_cost"))
        if mean_cost ≤ best_cost * (1 + tol)
            verbose && @info "collapsed"
            exit_status = :collapsed
            break
        end
        if mean_cost ≥ old_mean_cost * (1 - tol) && best_cost ≥ old_best_cost * (1 - tol)
            verbose && @info "cost didn't decrease by at least $tol, giving up"
            exit_status = :didntimprove
            break
        end
        old_mean_cost = mean_cost
        old_best_cost = best_cost
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

end # module
