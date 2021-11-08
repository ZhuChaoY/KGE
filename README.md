# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, TransR, TransD, ConvKB now, keep expanding) by tensorflow.

## Reference
(1) TransE: Translating Embeddings for Modeling Multi-relational Data   
(2) TransH: Knowledge Graph Embedding by Translating on Hyperplanes  
(3) TransR: Learning Entity and Relation Embeddings for Knowledge Graph Completion  
(4) TransD: Knowledge Graph Embedding via Dynamic Mapping Matrix  
(5) ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network  

## Operating Instructions
Run Run_KGE.py to train TransE, TransH, TransR, TransD, ConvKB.  
**TransX**:   
```
python Run_KGE.py --model TransE --dataset FB15k --dim 128 --margin 1.0 --l_r 1e-3 --batch_size 1024 --epoches 400 --do_train True --do_predict True
```
**ConvKB**:  
```
python Run_KGE.py --model ConvKB --dataset FB15k --dim 128 --n_filter 8 --l_r 1e-4 --batch_size 1024 --epoches 100 --do_train True --do_predict True
```

## Results (Filter)      
### FB15k (14951 E + 1345 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | **114.0** | 0.371 | 0.208 | 0.478 | 0.654 |
| **TransH** | 114.3 | 0.378 | 0.217 | **0.485** | 0.658 |
| **TransR** | 189.2 | 0.308 | 0.197 | 0.357 | 0.522 |
| **TransD** | 124.7 | 0.345 | 0.152 | 0.482 | **0.663** |
| **ConvKB** | 122.0 | **0.399** | **0.267** | 0.475 | 0.642 |

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 258.3 | 0.222 | 0.111 | 0.271 | 0.433 |
| **TransH** | **253.4** | 0.230 | 0.119 | 0.280 | 0.442 |
| **TransR** | 449.2 | 0.221 | 0.136 | 0.249 | 0.392 |
| **TransD** | 294.2 | 0.208 | 0.078 | 0.277 | **0.444** |
| **ConvKB** | 261.1 | **0.270** | **0.183** | **0.299** | **0.444** |

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 276.8 | 0.346 | 0.133 | 0.480 | 0.756 |
| **TransH** | **257.2** | 0.337 | 0.087 | 0.525 | 0.791 |
| **TransR** | 866.2 | 0.414 | **0.239** | 0.546 | 0.700 |
| **TransD** | 365.9 | **0.444** | 0.204 | **0.642** | **0.845** |
| **ConvKB** | 299.1 | 0.263 | 0.025 | 0.395 | 0.782 |

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | **3599.1** | 0.147 | 0.006 | 0.231 | 0.422 |
| **TransH** | 3605.9 | 0.156 | **0.011** | 0.254 | 0.430 |
| **TransR** | 8628.4 | 0.101 | 0.003 | 0.179 | 0.252 |
| **TransD** | 4589.5 | **0.165** | 0.007 | **0.278** | **0.446** |
| **ConvKB** | 4534.4 | 0.131 | 0.002 | 0.203 | 0.409 |
