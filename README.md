# R-GCN enhanced KGE
A Framework of R-GCN enhanced Knowledge Graph Embedding Models (Including TransE, TransH, TransD, ConvKB) by tensorflow.

## Main
(1) Pre-train KGE models by traditional process.   
(2) Serve a single layer of R-GCN as the encoder (The pre trained entity embeddings are the input feature of R-GCN), and a KGE model as the docoder, fine-tuning the pre-trained KGE models by few epoches.   

## Reference
(1) TransE: Translating Embeddings for Modeling Multi-relational Data   
(2) TransH: Knowledge Graph Embedding by Translating on Hyperplanes  
(3) TransD: Knowledge Graph Embedding via Dynamic Mapping Matrix  
(4) ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network    
(5) R-GCN: Modeling Relational Data with Graph Convolutional Networks  

## Results (Filter)      
### FB15k (14951 E + 1345 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 123.8 | 0.361 | 0.204 | 0.462 | 0.637 |
| **TransE (R-GCN)** | | | | | |
| **TransH** | **111.7** | 0.377 | 0.215 | **0.482** | 0.657 |
| **TransH (R-GCN)** | | | | | |
| **TransD** | 125.3 | 0.345 | 0.156 | 0.478 | **0.659** |
| **TransD (R-GCN)** | | | | | |
| **ConvKB** | 136.7 | **0.397** | **0.274** | 0.467 | 0.621 |
| **ConvKB (R-GCN)** | | | | | |

```
python Run_KGE.py --model TransX --dataset FB15k --margin 1.0
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --n_filter 16 --l_r 1e-4 --epoches 100
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 272.8 | 0.217 | 0.107 | 0.266 | 0.426 |
| **TransE (R-GCN)** | 281.8 | 0.232 | 0.135 | 0.269 | 0.420 |
| **TransH** | **266.5** | 0.224 | 0.111 | 0.276 | **0.433** |
| **TransH (R-GCN)** | 268.6 | 0.239 | 0.144 | 0.276 | 0.422 |
| **TransD** | 308.8 | 0.206 | 0.083 | 0.269 | 0.432 |
| **TransD (R-GCN)** | 278.8 | 0.226 | 0.119 | 0.275 | 0.427 |
| **ConvKB** | 288.4 | **0.253** | **0.162** | **0.288** | 0.427 |
| **ConvKB (R-GCN)** | | | | | |

```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0
```
```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0 --dropout 0.4 --l2 5e-3 --epoches 20 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 16 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 16 --dropout 0.4 --l2 5e-3 --epoches 20 --add_rgcn True
```

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 267.0 | 0.333 | 0.089 | 0.504 | 0.792 |
| **TransE (R-GCN)** | **251.3** | 0.398 | 0.183 | 0.551 | 0.789 |
| **TransH** | 290.1 | 0.336 | 0.090 | 0.512 | 0.793 |
| **TransH (R-GCN)** | 278.6 | 0.402 | 0.193 | 0.551 | 0.783 |
| **TransD** | 377.5 | 0.394 | 0.132 | **0.609** | **0.823** |
| **TransD (R-GCN)** | 335.3 | **0.435** | **0.214** | 0.605 | 0.811 |
| **ConvKB** | 300.4 | 0.223 | 0.034 | 0.300 | 0.645 |
| **ConvKB (R-GCN)** | 288.2 | 0.266 | 0.049 | 0.386 | 0.703 |

```
python Run_KGE.py --model TransX --dataset WN18 --margin 1.5
```
```
python Run_KGE.py --model TransX --dataset WN18 --margin 1.5 --dropout 0.4 --l2 5e-3 --epoches 20 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 16 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 16 --dropout 0.4 --l2 5e-3 --epoches 20 --add_rgcn True
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 3945.3 | 0.147 | 0.005 | 0.237 | 0.422 |
| **TransE (R-GCN)** | 3621.1 | 0.157 | 0.011 | 0.254 | 0.431 |
| **TransH** | 3860.3 | 0.153 | 0.006 | 0.250 | 0.429 |
| **TransH (R-GCN)** | 3471.4 | 0.159 | 0.011 | 0.257 | **0.437** |
| **TransD** | 4941.0 | 0.159 | 0.004 | **0.272** | 0.435 |
| **TransD (R-GCN)** | 4208.4 | **0.160** | 0.009 | **0.272** | 0.428 |
| **ConvKB** | 4454.9 | 0.139 | 0.002 | 0.227 | 0.403 |
| **ConvKB (R-GCN)** | **3446.2** | 0.131 | **0.018** | 0.180 | 0.378 |


```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5
```
```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5 --dropout 0.4 --l2 5e-3 --epoches 20 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 16 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 16 --dropout 0.4 --l2 5e-3 --epoches 20 --add_rgcn True
```
