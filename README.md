<!-- markdownlint-disable -->
<div id="top"></div>
<div align="center">
    <h1>tch-geometric</h1>
    <p>
        <a href="LICENSE">
            <img src="https://img.shields.io/github/license/EgorDm/tch-geometric" alt="License"></a>
        <a href="https://github.com/EgorDm/tch-geometric/actions/workflows/ci.yml">
             <img src="https://github.com/EgorDm/tch-geometric/actions/workflows/release.yml/badge.svg" alt="CI">
        </a>
        <a href="https://badge.fury.io/py/tch-geometric">
            <img src="https://badge.fury.io/py/tch-geometric.svg" alt="PyPI version">
        </a>
        <a href="https://anaconda.org/egordm/tch_geometric">
         <img src="https://anaconda.org/egordm/tch_geometric/badges/version.svg">
        </a>
    </p>
    <p>
        <b>Pytorch Geometric extension library</b>
    </p>
</div>
<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a>
</p>
<!-- markdownlint-enable -->

## Features
Pytorch Geometric extension library with additional graph sampling algorithms.

Supports:

* Node2Vec [^1] (`random_walk`)
* Temporal Random Walk (`temporal_random_walk`)
* Biased Temporal Random Walk (CTDNE) [^2] (`biased_tempo_random_walk`)
* Negative Sampling (`negative_sample_neighbors_homogenous` and `negative_sample_neighbors_heterogenous`)
* GraphSAGE + budget sampling (`budget_sampling`)
* Temporal Heterogenous Graph Transformer (HGT) sampling [^4] (`hgt_sampling`)
* GraphSAGE [^3] (`neighbor_sampling_heterogenous` and `neighbor_sampling_homogenous`)

[^1]: A. Grover and J. Leskovec, “node2vec: Scalable Feature Learning for Networks,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, New York, NY, USA, Aug. 2016, pp. 855–864. doi: 10.1145/2939672.2939754.
[^2]: G. H. Nguyen, J. B. Lee, R. A. Rossi, N. K. Ahmed, E. Koh, and S. Kim, “Continuous-Time Dynamic Network Embeddings,” in Companion of the The Web Conference 2018 on The Web Conference 2018 - WWW ’18, Lyon, France, 2018, pp. 969–976. doi: 10.1145/3184558.3191526.
[^3]: W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, Red Hook, NY, USA, Dec. 2017, pp. 1025–1035.
[^4] Z. Hu, Y. Dong, K. Wang, and Y. Sun, “Heterogeneous Graph Transformer (tempo),” arXiv:2003.01332 [cs, stat], Mar. 2020, Accessed: Mar. 09, 2022. [Online]. Available: http://arxiv.org/abs/2003.01332


## Installation
### CPU only
If you are using CPU only installation of `pytoch`, install tch-geometric using pip:
```
pip install tch_geometric
```

### GPU
If you are using GPU accelerated `pytoch` you need to use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
```
conda install egordm::tch_geometric
```

You can specify the cuda version explicity as follows (python 3.9, pytorch 1.11, cudnn 113):
```
conda install egordm::tch_geometric=0.1.0=py39_torch_1.11.0_cu113
```

## Examples
THe examples can be found in the [examples folder](examples/)
