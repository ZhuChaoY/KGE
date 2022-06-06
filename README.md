# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, ConvKB, RotatE) by tensorflow.  

## Reference
(1) **TransE**: [Translating Embeddings for Modeling Multi-relational Data](https://www.cs.sjtu.edu.cn/~li-fang/deeplearning-for-modeling-multi-relational-data.pdf)   
(2) **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf)   
(3) **ConvKB**: [A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network](https://arxiv.org/pdf/1712.02121.pdf)   
(4) **RotatE**: [ROTATE: KNOWLEDGE GRAPH EMBEDDING BY RELATIONAL ROTATION IN COMPLEX SPACE](https://arxiv.org/pdf/1902.10197.pdf) 

## Results 
(1) Fixed embedding dimension of **100**.  
(2) Only considering **filter** setting.  
(3) Early stopped by **Hist@10** reuslt on dev dataset.  

### FB15k (14951 E + 1345 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 77.4 | 0.599 | 0.466 | 0.696 | 0.817 |
| **TransH** | 77.1 | 0.606 | 0.474 | 0.705 | 0.820 |
| **ConvKB** | **72.0** | **0.618** | **0.492** | **0.712** | **0.825** |
| **RotatE** | 82.5 | 0.380 | 0.202 | 0.502 | 0.676 |

```
python Run_KGE.py --model [TransX] --dataset FB15k --margin 1.0 --l_r 5e-3 --batch_size 10000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --n_filter 50 --l_r 1e-3 --batch_size 10000 --epoches 500 --earlystop 5
```
```
python Run_KGE.py --model RotatE --dataset FB15k --margin 7.0 --l_r 5e-3 --batch_size 10000 --epoches 2000 --earlystop 2
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 294.0 | 0.314 | 0.224 | 0.351 | 0.489 |
| **TransH** | 286.0 | 0.326 | 0.238 | 0.363 | 0.497 |
| **ConvKB** | 253.2 | **0.334** | **0.245** | **0.371** | **0.509** |
| **RotatE** | **232.4** | 0.277 | 0.184 | 0.314 | 0.461 |

```
python Run_KGE.py --model [TransX] --dataset FB15k-237 --margin 1.0 --l_r 5e-3 --batch_size 10000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 50 --l_r 1e-3 --batch_size 10000 --epoches 500 --earlystop 5
```
```
python Run_KGE.py --model RotatE --dataset FB15k-237 --margin 8.0 --l_r 5e-3 --batch_size 10000 --epoches 2000 --earlystop 2
```

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 302.9 | 0.496 | 0.123 | 0.870 | 0.952 |
| **TransH** | 289.0 | 0.495 | 0.115 | 0.879 | **0.953** |
| **ConvKB** | 322.1 | 0.537 | 0.248 | 0.814 | 0.939 |
| **RotatE** | **193.7** | **0.917** | **0.900** | **0.928** | 0.946 |

```
python Run_KGE.py --model [TransX] --dataset WN18 --margin 4.0 --l_r 5e-3 --batch_size 5000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 50 --l_r 1e-3 --batch_size 5000 --epoches 500 --earlystop 5
```
```
python Run_KGE.py --model RotatE --dataset WN18 --margin 15.0 --l_r 5e-3 --batch_size 5000 --epoches 2000 --earlystop 2
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 3896.3 | 0.195 | 0.012 | 0.348 | 0.481 |
| **TransH** | 4033.7 | 0.195 | 0.011 | 0.346 | 0.476 |
| **ConvKB** | 4460.6 | 0.186 | 0.014 | 0.328 | 0.462 |
| **RotatE** | **3710.2** | **0.467** | **0.434** | **0.476** | **0.527** |

```
python Run_KGE.py --model [TransX] --dataset WN18RR --margin 4.0 --l_r 5e-3 --batch_size 5000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 50 --l_r 1e-3 --batch_size 5000 --epoches 500 --earlystop 5
```
```
python Run_KGE.py --model RotatE --dataset WN18RR --margin 15.0 --l_r 5e-3 --batch_size 5000 --epoches 2000 --earlystop 2
```

### Kinship (104 E + 25 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 6.5 | 0.452 | 0.230 | 0.611 | 0.846 |
| **TransH** | 5.0 | 0.530 | 0.321 | 0.684 | 0.901 |
| **ConvKB** | 3.3 | 0.717 | 0.594 | 0.804 | 0.939 |
| **RotatE** | **2.0** | **0.888** | **0.831** | **0.936** | **0.980** |

```
python Run_KGE.py --model [TransX] --dataset Kinship --margin 0.1 --l_r 5e-3 --batch_size 500 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset Kinship --n_filter 50 --l_r 1e-3 --batch_size 500 --epoches 500 --earlystop 5
```
```
python Run_KGE.py --model RotatE --dataset Kinship --margin 3.0 --l_r 5e-3 --batch_size 500 --epoches 2000 --earlystop 2
```

### UMLS (135 E + 46 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 1.6 | 0.859 | 0.764 | 0.947 | 0.989 |
| **TransH** | 1.6 | 0.862 | 0.766 | 0.954 | 0.990 |
| **ConvKB** | **1.3** | 0.918 | 0.849 | 0.987 | 0.994 |
| **RotatE** | **1.3** | **0.952** | **0.914** | **0.990** | **0.996** |

```
python Run_KGE.py --model [TransX] --dataset UMLS --margin 0.1 --l_r 5e-3 --batch_size 500 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset UMLS --n_filter 50 --l_r 1e-3 --batch_size 500 --epoches 500 --earlystop 5
```
```
python Run_KGE.py --model RotatE --dataset UMLS --margin 4.0 --l_r 5e-3 --batch_size 500 --epoches 2000 --earlystop 2
```

**[TransX]** from {TransE, TransH}  
