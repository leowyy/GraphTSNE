# GraphTSNE
**[Blog Post](https://leowyy.github.io/graphtsne)** | **[Paper](https://arxiv.org/abs/1904.06915)**

<b>GraphTSNE: A Visualization Technique for Graph-Structured Data</b> <br>
International Conference on Learning Representations 2019 <br>
Workshop for Representation Learning on Graphs and Manifolds <br>

<p align="center">
   <img src="pic/graphtsne.gif" width="500">
   <br>
   <b>GraphTSNE on the Cora Citation Network</b>
</p>

--------------------------------------------------------------------------------

## Codes
The code `demo_notebook.ipynb` creates a visualization of the Cora citation network using GraphTSNE. The original Cora dataset and other citation networks can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/.

The notebook takes roughly 3 minutes to run with GPU, or 8 minutes with CPU.
<br>

## Installation
   ```sh
   # Install Python libraries using conda
   conda env create -f environment.yml
   conda activate graph_tsne
   python -m ipykernel install --user --name graph_tsne --display-name "graph_tsne"

   # Run the notebook
   jupyter notebook
   ```
   
### When should I use this algorithm?
For visualizing graph-structured data such as social networks, functional brain networks and gene-regulatory networks. Concretely, graph-structured datasets contain two sources of information: graph connectivity between nodes and node features.

## Cite
If you use GraphTSNE in your work, we welcome you to cite our ICLR'19 workshop [paper](https://arxiv.org/abs/1904.06915): <br>
```
@inproceedings{leow19GraphTSNE,
  title={GraphTSNE: A Visualization Technique for Graph-Structured Data},
  author={Leow, Yao Yang and Laurent, Thomas and Bresson, Xavier},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019}
}
```
