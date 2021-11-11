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
| **TransE** | **104.9** | 0.371 | 0.198 | 0.487 | 0.667 |
| **TransH** | 108.9 | 0.378 | 0.215 | 0.486 | 0.661 |
| **TransD** | 117.2 | 0.346 | 0.145 | 0.492 | **0.675** |
| **ConvKB** | 122.3 | **0.426** | **0.295** | **0.504** | 0.661 |

```
python Run_KGE.py --model TransE --dataset FB15k --dim 512 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset FB15k --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset FB15k --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --dim 512 --n_filter 32 --l_r 1e-4 --epoches 100
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 247.7 | 0.223 | 0.106 | 0.277 | 0.444 |
| **TransH** | **241.2** | 0.230 | 0.116 | 0.283 | 0.444 |
| **TransD** | 291.4 | 0.205 | 0.072 | 0.278 | 0.448 |
| **ConvKB** | 253.5 | **0.288** | **0.202** | **0.318** | **0.459** |

```
python Run_KGE.py --model TransE --dataset FB15k-237 --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset FB15k-237 --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset FB15k-237 --dim 256 --margin 1.0 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --dim 256 --n_filter 8 --l_r 1e-4 --epoches 100
```

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 294.1 | 0.350 | 0.137 | 0.486 | 0.762 |
| **TransH** | 275.7 | 0.336 | 0.088 | 0.511 | 0.800 |
| **TransD** | 325.7 | **0.415** | **0.183** | **0.594** | **0.813** |
| **ConvKB** | **280.1** | 0.239 | 0.031 | 0.342 | 0.688 |

```
python Run_KGE.py --model TransE --dataset WN18 --dim 128 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset WN18 --dim 256 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset WN18 --dim 128 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --dim 256 --n_filter 32 --l_r 1e-4 --epoches 100
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | **3537.9** | 0.143 | 0.006 | 0.224 | 0.417 |
| **TransH** | 3656.1 | 0.154 | 0.006 | 0.255 | 0.431 |
| **TransD** | 4124.0 | **0.161** | 0.004 | **0.271** | **0.440** |
| **ConvKB** | 3760.2 | 0.149 | **0.019** | 0.220 | 0.412 |

```
python Run_KGE.py --model TransE --dataset WN18RR --dim 256 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransH --dataset WN18RR --dim 256 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model TransD --dataset WN18RR --dim 256 --margin 1.5 --l_r 1e-3 --epoches 500
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --dim 256 --n_filter 32 --l_r 1e-4 --epoches 100
```
