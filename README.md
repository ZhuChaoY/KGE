# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, TransR, TransD, ConvKB now, keep expanding)

## Files
### dataset/
FB15k (larger than 25MB, first unzip the file)  
FB15k-237  
WN18  
WN18RR  

**KGE.py** : Class of processing and tool functions for Knowledge Graph Embedding.  
**Models.py** : TransE, TransH, TransR, TransD, ConvKB structure.  
**Run_KGE.py** : Train and Predict KGE model.  

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
### FB15k
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** | 127.9 | 0.356 | 0.203 | 0.450 | 0.627 |  
|**TransH** | **118.8** | 0.375 | 0.229 | 0.462 | 0.635 |
|**TransR** | 193.3 | 0.282 | 0.174 | 0.326 | 0.495 |
|**TransD** | 133.8 | 0.351 | 0.188 | 0.454 | 0.635 |
|**ConvKB** | 134.9 | **0.393** | **0.267** | **0.463** | **0.624** |

### FB15k-237
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** | 127.9 | 0.356 | 0.203 | 0.450 | 0.627 |  
|**TransH** | **118.8** | 0.375 | 0.229 | 0.462 | 0.635 |
|**TransR** | 193.3 | 0.282 | 0.174 | 0.326 | 0.495 |
|**TransD** | 133.8 | 0.351 | 0.188 | 0.454 | 0.635 |
|**ConvKB** | 134.9 | **0.393** | **0.267** | **0.463** | **0.624** |

### WN18
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** | 127.9 | 0.356 | 0.203 | 0.450 | 0.627 |  
|**TransH** | **118.8** | 0.375 | 0.229 | 0.462 | 0.635 |
|**TransR** | 193.3 | 0.282 | 0.174 | 0.326 | 0.495 |
|**TransD** | 133.8 | 0.351 | 0.188 | 0.454 | 0.635 |
|**ConvKB** | 134.9 | **0.393** | **0.267** | **0.463** | **0.624** |

### WN18RR
|           | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|    --     |   --   |    --   |    --    |    --    |    --     |
|**TransE** | 127.9 | 0.356 | 0.203 | 0.450 | 0.627 |  
|**TransH** | **118.8** | 0.375 | 0.229 | 0.462 | 0.635 |
|**TransR** | 193.3 | 0.282 | 0.174 | 0.326 | 0.495 |
|**TransD** | 133.8 | 0.351 | 0.188 | 0.454 | 0.635 |
|**ConvKB** | 134.9 | **0.393** | **0.267** | **0.463** | **0.624** |
