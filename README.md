# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, ConvKB) by tensorflow.  

## Reference
(1) TransE: Translating Embeddings for Modeling Multi-relational Data   
(2) TransH: Knowledge Graph Embedding by Translating on Hyperplanes  
(3) ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network    

## Results 
(1) Fixed embedding dimension of **100**.  
(2) Only considering **filter** setting.  
(3) Early stopped by **Hist@10** reuslt on dev dataset.  

### FB15k (14951 E + 1345 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 85.0 | 0.575 | 0.446 | 0.667 | 0.790 |
| **TransH** | 82.6 | 0.580 | 0.452 | 0.670 | 0.793 |
| **ConvKB** | **79.2** | **0.596** | **0.470** | **0.686** | **0.805** |

```
python Run_KGE.py --model TransX --dataset FB15k --margin 1.0 --l_r 5e-3 --batch_size 10000 --epoches 5000
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --n_filter 50 --l_r 1e-3 --batch_size 10000 --epoches 500
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 299.3 | 0.312 | 0.225 | 0.347 | 0.482 |
| **TransH** | 300.0 | 0.318 | 0.230 | 0.355 | 0.488 |
| **ConvKB** | **251.7** | **0.330** | **0.242** | **0.365** | **0.500** |

```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0 --l_r 5e-3 --batch_size 10000 --epoches 5000
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 50 --l_r 1e-3 --batch_size 10000 --epoches 500
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
