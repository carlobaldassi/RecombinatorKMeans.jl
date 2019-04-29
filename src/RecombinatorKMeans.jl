module RecombinatorKMeans

using StatsBase
using SparseArrays
using Random
using Distributed

export kmeans, reckmeans

let cdict = Dict{NTuple{3,Int},Array{Float64,3}}()
    global function get_costs!(c, data, centroids)
        m, n = size(data)
        k = size(centroids, 2)
        @assert size(centroids, 1) == m

        cache = get!(cdict, (m, k, n)) do
            zeros(m, k, n)
        end

        cache .= (reshape(data, (m, 1, n)) .- centroids).^ 2
        costs, cc = findmin(dropdims(sum(cache, dims=1), dims=1), dims=1)
        @assert all(x->1 ≤ x[1] ≤ k, vec(cc))
        c .= map(x->x[1], vec(cc))
        return costs
    end

    global function assign_points!(c, data, centroids)
        k = size(centroids, 2)

        costs = get_costs!(c, data, centroids)
        return sum(costs)
    end
    global function clear_cache!()
        empty!(cdict)
    end
    global function getdict()
        return cdict
    end
end

function recompute_centroids!(c, data, centroids)
    k = size(centroids, 2)
    # TODO: optimize
    for j = 1:k
        msk = c .== j
        any(msk) || continue
        centroids[:,j] .= vec(mean(data[:,msk], dims=2))
    end
    return centroids
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
    centr[:,1] .= pool[:,i]

    costs = sum((data .- centr[:,1]).^2, dims=1)
    pcosts = dataispool ? costs : sum((pool .- centr[:,1]).^2, dims=1)

    curr_cost = sum(costs)
    c = ones(Int, n)

    new_costs, new_c = similar(costs), similar(c)
    new_costs_best, new_c_best = similar(costs), similar(c)
    new_pcosts_best = dataispool ? new_costs_best : similar(pcosts)
    for j = 2:k
        candidates = [sample(1:np, Weights(vec(pcosts) .* w)) for _ = 1:ncandidates]
        cost_best = Inf
        i_best = 0
        for i in candidates
            new_c .= c
            new_costs .= sum((data .- pool[:,i]).^2, dims=1)
            for l = 1:n
                if new_costs[l] < costs[l]
                    new_c[l] = j
                else
                    new_costs[l] = costs[l]
                end
            end
            cost = sum(new_costs)
            if cost < cost_best
                cost_best = cost
                i_best = i
                new_costs_best .= new_costs
                if !dataispool
                    new_pcosts_best .= min.(pcosts, sum((pool .- pool[:,i]).^2, dims=1))
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

function kmeans(data::Matrix{Float64}, k::Integer;
                max_it::Integer = 1000,
                seed::Integer = 1324235434345,
                init::Union{String,Matrix{Float64}} = "++",
                verbose::Bool = true
               )
    Random.seed!(seed)
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

function reckmeans(data::Matrix{Float64}, k::Integer, Rlist;
                   β = 10.0,
                   seed::Integer = 1324235434345,
                   tol::Float64 = 1e-4,
                   verbose::Bool = true,
                   keepalllosses::Bool = false
                  )
    Random.seed!(seed)
    m, n = size(data)

    best_cost = Inf
    best_centr = Matrix{Float64}(undef, m, k)

    dd = copy(data)
    w = ones(size(dd, 2))
    old_mean_cost = Inf
    old_best_cost = Inf
    allcosts = keepalllosses ? Vector{Float64}[] : nothing
    exit_status = :running
    if Rlist isa Int
        Rlist = Iterators.repeated(Rlist)
    end
    for (it,R) in enumerate(Rlist)
        verbose && @info "it = $it R = $R"
        @assert length(w) == size(dd, 2)
        h0 = hash(dd)
        res = pmap(1:R) do a
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
        resize!(w, R*k)
        mean_cost = mean(costs)
        for a = 1:R
            w[(1:k) .+ (a-1)*k] .= exp(-β * ((costs[a] - batch_best_cost) / (mean_cost - batch_best_cost)))
        end
        w ./= sum(w)
        dd = hcat(centroidsR...)
        verbose && println("  mean cost = $mean_cost best_cost = $best_cost old_best_cost = $old_best_cost")
        if (mean_cost / best_cost - 1) ≤ tol
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
