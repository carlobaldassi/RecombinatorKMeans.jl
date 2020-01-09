# RecombinatorKMeans.jl

This code implements the recombinator-k-means method described in the paper
&"Recombinator-k-means: A population based algorithm that exploits k-means++ for recombination"* by C. Baldassi
submitted for publication, (2019) ([arXiv][RKMarXiv]).

The code is written in [Julia]. It requires Julia 1.0 or later.

This code works fine and it's usable, but it is intended as a demo and a reference implementation.
For this reason, it has a few limitations, the main one being that it is not flexible or generic:
it only works with data stored in dense `Float64` matrices, and it only uses the squared Euclidean
distance as a metric. It also tries to reduce the number of options at a minimum. It's also
somewhat liberal in terms of memory usage (particularly if you run it in parallel).

It provides three main optimization methods, which are exported from the package:

* `kmeans` is a standard implementation of Lloyd's algorithm for k-means; it can use either uniform
  of k-means++ initialization (the latter in the improved version that is also used by scikit-learn)
* `reckmeans` is the recombinator-k-means method described in the paper
* `kmeans_randswap` is the random swap algorithm proposed in [this paper](https://link.springer.com/article/10.1186/s40537-018-0122-y)

It also provides two functions to compute the centroid index as defined in
[this paper](https://link.springer.com/article/10.1186/s40537-018-0122-y), an
asymmetric one called `CI` and a symmetric one called `CI_sym`. These are not exported.

### Installation and setup

To install the module, just clone it from GitHub into some directory. Then enter in such directory
and run julia with the "project" option:

```
$ julia --project
```

(Alternatively, if you start Julia from some other directory, you can press <kbd>;</kbd> to enter
in shell mode, `cd` into the project's directory, enter in pkg mode with <kbd>]</kbd> and use the
`activate` command.)

The first time you do this, you will then need to setup the project's environment. To do that,
when you're in the Julia REPL, press the <kbd>]</kbd> key to enter in pkg mode, then resolve the
dependencies:

```
(RecombinatorKMeans) pkg> resolve
```

This should download all the required packages. You can subsequently type `test` to check that
everything works. After this, you can press the backspace key to get back to the standard Julia
prompt, and load the package:

```
julia> using RecombinatorKMeans
```

### Usage

The format of the data must be a `Matrix{Float64}` with the data points organized by column.
(Typically, this means that if you're reading a dataset you'll need to transpose it. See for
example the `runfile.jl` script in the `test` directory.)

These three functions are available once you load the package: `kmeans`, `reckmeans` and `kmeans_randswap`. You
can use the Julia help (press the <kbd>?</kbd> key in the REPL) to see their documentation.

The `reckmeans` function will run in parallel if there are workers available. However, the code
must be loaded on the workers too. To do this, run Julia with the `p` option:

```
$ julia -p 4 # this will use 4 cores
```

Then, before loading the package, do the following at the REPL:

```
julia> @everywhere using Pkg
julia> @everywhere Pkg.activate(".")
```

(This assumes that you are running in the project's main directory, otherwise change `"."` to
the correct path.)

After this `using RecombinatorKMeans` should work and `reckmeans` should run in parallel.

### Reproducing the results in the paper

For the purpose of complete reproducibility, you can check out the tag `paper-v2b` of the repository,
which will get you the version of the code used to collect the results in the [paper][RKMarXiv].
Also, the repository includes a file "Manifest_20191123.toml" that specifies the exact version of the
dependencies that were used. You can use it to overwrite your "Manifest.toml" file and then call
`instantiate` in pkg mode to reproduce the same environment.

## Licence

The code is released under the MIT licence.

The k-means++ code was first written from scratch from [the k-means++ paper][km++], then improved after reading
the corresponding [scikit-learn's code][sklearnkmeans], then heavily modified.
The scikit-learn's version was first coded by Jan Schlueter as a port of some other code that is now lost.

[Julia]: https://julialang.org
[RKMarXiv]: https://arxiv.org/abs/1905.00531
[km++]: https://scholar.google.com/scholar?cluster=16794944444927209316
[sklearnkmeans]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/k_means_.py
