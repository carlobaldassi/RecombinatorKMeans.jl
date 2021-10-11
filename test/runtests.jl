using RecombinatorKMeans
using Test
using DelimitedFiles

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
    labels, centroids, cost, converged = kmeans(a3, k, init="unif", verbose=false)
    @test length(labels) == m
    @test all(∈(1:k), labels)
    @test size(centroids) == (2,k)
    @test 7 < cost < 25
    @test converged == true
end

@testset "kmeans++" begin
    labels, centroids, cost, converged = kmeans(a3, k, init="++", verbose=false)
    @test length(labels) == m
    @test all(∈(1:k), labels)
    @test size(centroids) == (2,k)
    @test 6.7 < cost < 11
    @test converged == true
end

@testset "reckmeans" begin
    res = reckmeans(a3, k, 5, Δβ = 0.1, verbose=false)
    @test res.exit_status == :collapsed
    @test length(res.labels) == m
    @test all(∈(1:k), res.labels)
    @test size(res.centroids) == (2,k)
    @test 6.7 < res.cost < 7.5
end

@testset "gakmeans" begin
    res = gakmeans(a3, k, 5, verbose=false)
    @test res.exit_status == :collapsed
    @test length(res.labels) == m
    @test all(∈(1:k), res.labels)
    @test size(res.centroids) == (2,k)
    @test 6.7 < res.cost < 7.5
end
