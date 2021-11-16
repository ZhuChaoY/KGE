# R-GCN enhanced KGE
A Framework of R-GCN enhanced Knowledge Graph Embedding Models (Including TransE, TransH, ConvKB) by tensorflow.

## Main
(1) Pre-train KGE models by traditional process.   
(2) Serve a single layer of R-GCN as the encoder (The pre trained entity embeddings are the input feature of R-GCN), and a KGE model as the docoder, fine-tuning the pre-trained KGE models by few epoches.   

## Reference
(1) TransE: Translating Embeddings for Modeling Multi-relational Data   
(2) TransH: Knowledge Graph Embedding by Translating on Hyperplanes  
(3) ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network    
(4) R-GCN: Modeling Relational Data with Graph Convolutional Networks  

## Results 
(1) Fixed embedding dimension of **100**.  
(2) Only considering **filter** setting.  
(3) Early stopped by **Hist@10** reuslt on dev dataset.  

### FB15k (14951 E + 1345 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 154.1 | 0.395 | 0.258 | 0.476 | 0.644 |
| **TransE (R-GCN)** | | | | | |
| **TransH** | 140.1 | 0.410 | 0.275 | 0.491 | 0.657 |
| **TransH (R-GCN)** | | | | | |
| **ConvKB** | | | | | |
| **ConvKB (R-GCN)** | | | | | |

```
python Run_KGE.py --model TransX --dataset FB15k --margin 1.0 --n_neg 2 --l_r 5e-3
```
```
python Run_KGE.py --model TransX --dataset FB15k --margin 1.0 --n_neg 2 --dropout 0.4 --epoches 100 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --n_filter 64 --n_neg 2
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --n_filter 64 --n_neg 2 --dropout 0.4 --epoches 100 --add_rgcn True
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 299.8 | 0.249 | 0.152 | 0.287 | 0.441 |
| **TransE (R-GCN)** | | | | | |
| **TransH** | 284.8 | 0.262 | 0.167 | 0.299 | 0.449 |
| **TransH (R-GCN)** | | | | | |
| **ConvKB** | | | | | |
| **ConvKB (R-GCN)** | | | | | |

```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0 --n_neg 2 --l_r 5e-3
```
```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0 --n_neg 2 --dropout 0.4 --epoches 100 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 64 --n_neg 2
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 64 --n_neg 2 --dropout 0.4 --epoches 100 --add_rgcn True
```

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 230.0 | 0.422 | 0.230 | 0.547 | 0.794 |
| **TransE (R-GCN)** | | | | | |
| **TransH** | 204.4 | 0.434 | 0.225 | 0.591 | 0.807 |
| **TransH (R-GCN)** | | | | | |
| **ConvKB** | | | | | |
| **ConvKB (R-GCN)** | | | | | |

```
python Run_KGE.py --model TransX --dataset WN18 --margin 1.5 --n_neg 4 --l_r 5e-3
```
```
python Run_KGE.py --model TransX --dataset WN18 --margin 1.5 --n_neg 4 --dropout 0.4 --epoches 100 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 64 --n_neg 4
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 64 --n_neg 4 --dropout 0.4 --epoches 100 --add_rgcn True
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 3130.7 | 0.168 | 0.029 | 0.245 | 0.453 |
| **TransE (R-GCN)** | | | | | |
| **TransH** | 2862.4 | 0.194 | 0.030 | 0.316 | 0.473 |
| **TransH (R-GCN)** | | | | | |
| **ConvKB** | | | | | |
| **ConvKB (R-GCN)** | | | | | |


```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5 --n_neg 4 --l_r 5e-3
```
```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5 --n_neg 4 --dropout 0.4 --epoches 100 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 64 --n_neg 4
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 64 --n_neg 4 --dropout 0.4 --epoches 100 --add_rgcn True
```
