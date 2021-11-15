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

## Results 
(1) Fixed embedding dimension of **100**.  
(2) Only consider **filter** setting.  
(3) Early stopped by **Hist@10** reuslt on dev dataset.  

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
| **TransH** | 266.5 | 0.224 | 0.111 | 0.276 | **0.433** |
| **TransH (R-GCN)** | 268.6 | 0.239 | 0.144 | 0.276 | 0.422 |
| **TransD** | 308.8 | 0.206 | 0.083 | 0.269 | 0.432 |
| **TransD (R-GCN)** | 278.8 | 0.226 | 0.119 | 0.275 | 0.427 |
| **ConvKB** | 288.4 | 0.253 | 0.162 | **0.288** | 0.427 |
| **ConvKB (R-GCN)** | **245.6** | **0.261** | **0.176** | **0.288** | 0.429 |

```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0
```
```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0 --dropout 0.4 --epoches 20 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 16 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 16 --dropout 0.4 --epoches 20 --add_rgcn True
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
python Run_KGE.py --model TransX --dataset WN18 --margin 1.5 --dropout 0.4 --epoches 20 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 16 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 16 --dropout 0.4 --epoches 20 --add_rgcn True
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 3334.2 | 0.149 | 0.008 | 0.230 | 0.418 |
| **TransE (R-GCN)** | 3279.3 | 0.152 | 0.005 | 0.252 | 0.414 |
| **TransH** | 3221.7 | **0.159** | 0.007 | **0.264** | **0.433** |
| **TransH (R-GCN)** | **3028.8** | 0.152 | 0.005 | 0.253 | 0.415 |
| **TransD** | 5221.6 | 0.153 | 0.008 | 0.246 | 0.421 |
| **TransD (R-GCN)** | 3699.9 | 0.155 | 0.005 | 0.255 | 0.425 |
| **ConvKB** | 3645.0 | 0.154 | 0.016 | 0.239 | 0.417 |
| **ConvKB (R-GCN)** | 3284.4 | 0.145 | **0.030** | 0.194 | 0.396 |


```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5 --l_r 1e-2
```
```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5 --dropout 0.4 --epoches 20 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 16 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 16 --dropout 0.4 --epoches 20 --add_rgcn True
```
