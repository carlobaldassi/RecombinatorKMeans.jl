using RecombinatorKMeans
using Test
using DelimitedFiles
using Distributed

# The a3 dataset was downloaded from here:
#
#  http://cs.uef.fi/sipu/datasets/
#
#  Clustering basic benchmark
#  P. Fränti and S. Sieranoja
#  K-means properties on six clustering benchmark datasets
#  Applied Intelligence, 48 (12), 4743-4759, December 2018
#  https://doi.org/10.1007/s10489-018-1238-7

a3 = Matrix(readdlm(joinpath(@__DIR__, "a3.txt"))')
a3 ./= maximum(a3)
n, m = size(a3)
k = 50

@testset "kmeans uniform" begin
    labels, centroids, cost = kmeans(a3, k, init="unif", verbose=false)
    @test length(labels) == m
    @test all(∈(1:k), labels)
    @test size(centroids) == (2,k)
    @test 7 < cost < 25
end

@testset "kmeans++" begin
    labels, centroids, cost = kmeans(a3, k, init="++", verbose=false)
    @test length(labels) == m
    @test all(∈(1:k), labels)
    @test size(centroids) == (2,k)
    @test 6.7 < cost < 11
end

@testset "reckmeans" begin
    res = reckmeans(a3, k, 5, β = 5.0, verbose=false)
    @test res.exit_status ∈ [:collapsed, :didntimprove]
    @test length(res.labels) == m
    @test all(∈(1:k), res.labels)
    @test size(res.centroids) == (2,k)
    @test 6.7 < res.cost < 7.5
    @test res.all_costs ≡ nothing

    res = reckmeans(a3, k, 5, β = 5.0, verbose=false, keepallcosts=true)
    @test res.exit_status ∈ [:collapsed, :didntimprove]
    @test length(res.labels) == m
    @test all(∈(1:k), res.labels)
    @test size(res.centroids) == (2,k)
    @test 6.7 < res.cost < 7.5
    @test res.all_costs isa Vector{Vector{Float64}}
    @test all(res.all_costs) do vl
        all(6.7 .< vl .< 11)
    end
end

@testset "reckmeans parallel" begin
    addwrk = addprocs(2)
    @everywhere begin
        using Pkg
        Pkg.activate(joinpath(@__DIR__, ".."))
        using RecombinatorKMeans
    end
    res = reckmeans(a3, k, 5, β = 5.0, verbose=false)
    @test res.exit_status ∈ [:collapsed, :didntimprove]
    @test length(res.labels) == m
    @test all(∈(1:k), res.labels)
    @test size(res.centroids) == (2,k)
    @test 6.7 < res.cost < 7.5
    @test res.all_costs ≡ nothing
    rmprocs(addwrk...)
end
