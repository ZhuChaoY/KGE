# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, TransR, TransD, ConvKB now, keep expanding)

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

## Results       
### FB15k (14951 E + 1345 R)
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** |  |  |  | |  |  
|**TransH** |  |  |  |  |  |
|**TransR** |  |  |  |  |  |
|**TransD** |  |  |  |  | |
|**ConvKB** | |  |  |  |  |

### FB15k-237 (14541 E + 237 R)
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** | 271.7 | 0.225 | 0.122 | 0.267 | 0.424 |  
|**TransH** | 263.8 | 0.226 | 0.117 | 0.272 | 0.433 |
|**TransR** |  |  |  |  |  |
|**TransD** | 316.3 | 0.204 | 0.086 | 0.262 | 0.423 |
|**ConvKB** |  |  |  |  |  |

### WN18 (40943 E + 18 R)
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** | **249.0** | 0.332 | 0.117 | 0.469 | 0.748 |  
|**TransH** | 291.8 | 0.337 | 0.032 | 0.588 | **0.858** |
|**TransR** | 527.5 | 0.302 | **0.156** | 0.386 | 0.576 |
|**TransD** | 365.2 | **0.352** | 0.054 | **0.600** | 0.839 |
|**ConvKB** |  |  |  |  |  |

### WN18RR (40943 E + 11 R)
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** | **3144.9** | 0.149 | 0.012 | 0.221 | 0.416 |  
|**TransH** | 3401.8 | **0.165** | 0.011 | **0.273** | **0.438** |
|**TransR** | 5703.8 | 0.122 | 0.003 | 0.210 | 0.313 |
|**TransD** | 4232.1 | 0.159 | 0.008 | 0.259 | 0.436 |
|**ConvKB** | 4181.2 | 0.126 | **0.018** | 0.175 | 0.348 |
