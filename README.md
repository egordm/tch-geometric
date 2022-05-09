<!-- markdownlint-disable -->
<div id="top"></div>
<div align="center">
    <h1>tch-geometric</h1>
    <p>
       <a href="LICENSE"><img src="https://img.shields.io/github/license/EgorDm/nauman" alt="License"></a>
 <a href="https://github.com/EgorDm/tch-geometric/actions/workflows/ci.yml"><img src="https://github.com/EgorDm/tch-geometric/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
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

* Node2Vec (`random_walk`)
* Temporal Random Walk (`temporal_random_walk`)
* Biased Temporal Random Walk (CTDNE) (`biased_tempo_random_walk`)
* Negative Sampling (`negative_sample_neighbors_homogenous` and `negative_sample_neighbors_heterogenous`)
* GraphSAGE budget sampling (`budget_sampling`)
* Temporal Heterogenous Graph Transformer (HGT) sampling (`hgt_sampling`)
* GraphSAGE (`neighbor_sampling_heterogenous` and `neighbor_sampling_homogenous`)

## TODO:
* Cite appropriately
* Add usage guide


## Examples
Check examples folder