# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, ConvKB) by tensorflow.  

## Reference
(1) **TransE**: [Translating Embeddings for Modeling Multi-relational Data](https://www.cs.sjtu.edu.cn/~li-fang/deeplearning-for-modeling-multi-relational-data.pdf)   
(2) **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf)   
(3) **ConvKB**: [A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network](https://arxiv.org/pdf/1712.02121.pdf)   

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
python Run_KGE.py --model [TransX] --dataset FB15k --margin 1.0 --l_r 5e-3 --batch_size 10000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --n_filter 50 --l_r 1e-3 --batch_size 10000 --epoches 500 --earlystop 5
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 299.3 | 0.312 | 0.225 | 0.347 | 0.482 |
| **TransH** | 300.0 | 0.318 | 0.230 | 0.355 | 0.488 |
| **ConvKB** | **251.7** | **0.330** | **0.242** | **0.365** | **0.500** |

```
python Run_KGE.py --model [TransX] --dataset FB15k-237 --margin 1.0 --l_r 5e-3 --batch_size 10000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 50 --l_r 1e-3 --batch_size 10000 --epoches 500 --earlystop 5
```

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | **290.0** | 0.462 | 0.106 | 0.808 | **0.937** |
| **TransH** | 302.0 | 0.465 | 0.105 | **0.817** | 0.936 |
| **ConvKB** | 300.2 | **0.499** | **0.226** | 0.743 | 0.911 |

```
python Run_KGE.py --model [TransX] --dataset WN18 --margin 4.0 --l_r 5e-3 --batch_size 3000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 50 --l_r 1e-3 --batch_size 3000 --epoches 500 --earlystop 5
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 3848.1 | 0.185 | 0.012 | 0.319 | 0.471 |
| **TransH** | **3795.8** | **0.188** | 0.011 | **0.330** | **0.474** |
| **ConvKB** | 4317.9 | 0.153 | **0.018** | 0.236 | 0.415 |

```
python Run_KGE.py --model [TransX] --dataset WN18RR --margin 4.0 --l_r 5e-3 --batch_size 3000 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 50 --l_r 1e-3 --batch_size 3000 --epoches 500 --earlystop 5
```

### Kinship (104 E + 25 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 6.7 | 0.463 | 0.258 | 0.607 | 0.834 |
| **TransH** | **5.3** | **0.543** | 0.334 | **0.711** | **0.878** |
| **ConvKB** |  |  |  |  |  |

```
python Run_KGE.py --model [TransX] --dataset Kinship --margin 0.1 --l_r 5e-3 --batch_size 500 --epoches 5000 --earlystop 2
```
```
python Run_KGE.py --model ConvKB --dataset Kinship --n_filter 50 --l_r 1e-3 --batch_size 500 --epoches 500 --earlystop 5
```

**[TransX]** from {TransE, TransH}

