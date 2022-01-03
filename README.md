# RecombinatorKMeans.jl

This code implements the recombinator-k-means method described in the paper
*"Recombinator-k-means: An evolutionary algorithm that exploits k-means++ for recombination"* by C. Baldassi
submitted for publication, (2019) ([arXiv][RKMarXiv]).

The code is written in [Julia]. It requires Julia 1.6 or later.

It provides four main optimization methods, which are exported from the package:

* `kmeans` is a standard implementation of Lloyd's algorithm for k-means; it can use either uniform
  of k-means++ initialization (the latter in the improved version that is also used by scikit-learn)
* `reckmeans` is the recombinator-k-means method described in the paper
* `gakmeans` is the genetic algorithm with pairwise-nearest-neighbor crossover proposed in [this paper][GA]
* `rswapkmeans` is the random-swap algorithm proposed in [this paper][RS]

It also provides two functions to compute the centroid index as defined in [this paper][CI], an
asymmetric one called `CI` and a symmetric one called `CI_sym`. These are not exported.

It also provides a function to compute the variation of information metric to quantify the
distance between two partitions as defined in [this paper][VI]. The function is called `VI` and is
not exported.

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

These four functions are available once you load the package: `kmeans`, `reckmeans`, `gakmeans` and
`rswapkmeans`. You can use the Julia help (press the <kbd>?</kbd> key in the REPL) to see their
documentation.

The `reckmeans` and `gakmeans` functions will run in parallel if there are threads available:
either run Julia with the `-t` option or use the `JULIA_NUM_THREADS` environment variable.

### Reproducing the results in the paper

For the purpose of complete reproducibility, you can check out the tag `paper-v5` of the repository,
which will get you the version of the code used to collect the results in the [paper][RKMarXiv].
Also, the repository includes a file "Manifest_20220103.toml" that specifies the exact version of
the dependencies that were used. You can use it to overwrite your "Manifest.toml" file and then
call `instantiate` in pkg mode to reproduce the same environment. Note that the version of Julia
should be the same as that in the paper too.

## Licence

The code is released under the MIT licence.

The k-means++ code was first written from scratch from [the k-means++ paper][km++], then improved
after reading the corresponding [scikit-learn's code][sklearnkmeans], then heavily modified.  The
scikit-learn's version was first coded by Jan Schlueter as a port of some other code that is now
lost.

The genetic algorithm code was written from scratch from [the paper][GA]; the accompanying C code
available at [the repository][GAcode] was inspected to check some finer details of the behavior, but
none of the code was used.

[Julia]: https://julialang.org
[RKMarXiv]: https://arxiv.org/abs/1905.00531
[km++]: https://scholar.google.com/scholar?cluster=16794944444927209316
[sklearnkmeans]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/_kmeans.py
[GA]: https://www.sciencedirect.com/science/article/abs/pii/S0167865599001336
[RS]: https://link.springer.com/article/10.1186/s40537-018-0122-y
[GAcode]: https://archive.uef.fi/en/web/machine-learning/software/
[CI]: https://www.sciencedirect.com/science/article/abs/pii/S0031320314001150
[VI]: https://www.sciencedirect.com/science/article/pii/S0047259X06002016?via%3Dihub
