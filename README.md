# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, TransD, ConvKB) by tensorflow.

## Reference
(1) TransE: Translating Embeddings for Modeling Multi-relational Data   
(2) TransH: Knowledge Graph Embedding by Translating on Hyperplanes  
(3) TransD: Knowledge Graph Embedding via Dynamic Mapping Matrix  
(4) ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network  

## Results (Filter)      
### FB15k (14951 E + 1345 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 111.2 | 0.378 | 0.210 | 0.491 | 0.668 |
| **TransH** | **109.7** | 0.384 | 0.217 | 0.495 | **0.671** |
| **TransD** | 124.5 | 0.351 | 0.158 | 0.490 | **0.671** |
| **ConvKB** | 122.5 | **0.421** | **0.290** | **0.497** | 0.656 |

```
python Run_KGE.py --model TransE --dataset FB15k --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset FB15k --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset FB15k --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --dim 256 --n_filter 8 --l_r 1e-4 --epoches 100
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 252.8 | 0.222 | 0.108 | 0.272 | 0.436 |
| **TransH** | **242.8** | 0.231 | 0.117 | 0.284 | 0.444 |
| **TransD** | 291.9 | 0.212 | 0.088 | 0.274 | 0.440 |
| **ConvKB** | 260.3 | **0.284** | **0.199** | **0.314** | **0.454** |

```
python Run_KGE.py --model TransE --dataset FB15k-237 --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset FB15k-237 --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset FB15k-237 --dim 128 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --dim 256 --n_filter 32 --l_r 1e-4 --epoches 100
```

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 276.6 | 0.350 | 0.131 | 0.490 | 0.775 |
| **TransH** | **251.9** | 0.345 | 0.123 | 0.496 | 0.773 |
| **TransD** | 339.7 | **0.492** | **0.271** | **0.674** | **0.866** |
| **ConvKB** | 292.4 | 0.248 | 0.048 | 0.338 | 0.689 |

```
python Run_KGE.py --model TransE --dataset WN18 --dim 128 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset WN18 --dim 128 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset WN18 --dim 256 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --dim 256 --n_filter 32 --l_r 1e-4 --epoches 100
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 2768.7 | 0.141 | **0.035** | 0.175 | 0.374 |
| **TransH** | **2458.0** | 0.141 | 0.034 | 0.175 | 0.382 |
| **TransD** | 4404.1 | **0.161** | 0.004 | **0.279** | **0.439** |
| **ConvKB** | 3542.6 | 0.135 | 0.015 | 0.190 | 0.396 |

```
python Run_KGE.py --model TransE --dataset WN18RR --dim 256 --margin 2.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset WN18RR --dim 256 --margin 2.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset WN18RR --dim 256 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --dim 256 --n_filter 32 --l_r 1e-4 --epoches 100
```
