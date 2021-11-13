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

## Results (Filter)      
### FB15k (14951 E + 1345 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 113.2 | 0.372 | 0.207 | 0.480 | 0.658 |
| **TransE (R-GCN)** | | | | | |
| **TransH** | **108.9** | 0.378 | 0.215 | 0.486 | 0.661 |
| **TransH (R-GCN)** | | | | | |
| **TransD** | 117.2 | 0.346 | 0.145 | **0.492** | **0.675** |
| **TransD (R-GCN)** | | | | | |
| **ConvKB** | 120.6 | **0.416** | **0.286** | **0.492** | 0.652 |
| **ConvKB (R-GCN)** | | | | | |

```
python Run_KGE.py --model TransX --dataset FB15k --margin 1.0
```
```
python Run_KGE.py --model ConvKB --dataset FB15k --n_filter 8 --l_r 1e-4 --epoches 100
```

### FB15k-237 (14541 E + 237 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 247.7 | 0.223 | 0.106 | 0.277 | 0.444 |
| **TransE (R-GCN)** | 260.2 | 0.242 | 0.143 | 0.282 | 0.432 |
| **TransH** | 241.2 | 0.230 | 0.116 | 0.283 | 0.444 |
| **TransH (R-GCN)** | 273.8 | 0.238 | 0.141 | 0.278 | 0.428 |
| **TransD** | 291.4 | 0.205 | 0.072 | 0.278 | 0.448 |
| **TransD (R-GCN)** | 258.7 | 0.231 | 0.120 | 0.282 | 0.438 |
| **ConvKB** | 253.5 | **0.288** | **0.202** | **0.318** | **0.459** |
| **ConvKB (R-GCN)** | **232.6** | 0.270 | 0.184 | 0.299 | 0.443 |

```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0
```
```
python Run_KGE.py --model TransX --dataset FB15k-237 --margin 1.0 --dropout 0.4 --l2 5e-3 --epoches 20 --earlystop 3 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 8 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset FB15k-237 --n_filter 8 --dropout 0.4 --l2 5e-3 --epoches 20 --earlystop 3 --add_rgcn True
```

### WN18 (40943 E + 18 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 301.7 | 0.331 | 0.087 | 0.499 | 0.798 |
| **TransE (R-GCN)** | 271.4 | 0.400 | 0.193 | 0.542 | 0.787 |
| **TransH** | 275.7 | 0.336 | 0.088 | 0.511 | 0.800 |
| **TransH (R-GCN)** | **266.2** | 0.402 | 0.185 | 0.559 | 0.791 |
| **TransD** | 351.7 | 0.405 | 0.140 | **0.625** | **0.842** |
| **TransD (R-GCN)** | 308.6 | **0.440** | **0.218** | 0.612 | 0.820 |
| **ConvKB** | 280.1 | 0.239 | 0.031 | 0.342 | 0.688 |
| **ConvKB (R-GCN)** | 288.5 | 0.283 | 0.058 | 0.415 | 0.733 |

```
python Run_KGE.py --model TransX --dataset WN18 --margin 1.5
```
```
python Run_KGE.py --model TransX --dataset WN18 --margin 1.5 --dropout 0.4 --l2 5e-3 --epoches 20 --earlystop 3 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 32 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset WN18 --n_filter 32 --dropout 0.4 --l2 5e-3 --epoches 20 --earlystop 3 --add_rgcn True
```

### WN18RR (40943 E + 11 R)
|            | **MR** | **MRR** |**Hist@1**|**Hist@3**|**Hist@10**|
|     --     |   --   |    --   |    --    |    --    |    --     |
| **TransE** | 3537.9 | 0.143 | 0.006 | 0.224 | 0.417 |
| **TransE (R-GCN)** | **3185.4** | 0.156 | 0.009 | 0.248 | 0.437 |
| **TransH** | 3656.1 | 0.154 | 0.006 | 0.255 | 0.431 |
| **TransH (R-GCN)** | 3502.1 | 0.156 | 0.010 | 0.252 | 0.437 |
| **TransD** | 4124.0 | 0.161 | 0.004 | 0.271 | **0.440** |
| **TransD (R-GCN)** | 4031.7 | **0.165** | 0.006 | **0.286** | 0.436 |
| **ConvKB** | 3760.2 | 0.149 | 0.019 | 0.220 | 0.412 |
| **ConvKB (R-GCN)** | 3477.9 | 0.160 | **0.042** | 0.218 | 0.411 |


```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5
```
```
python Run_KGE.py --model TransX --dataset WN18RR --margin 1.5 --dropout 0.4 --l2 5e-3 --epoches 20 --earlystop 3 --add_rgcn True
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 32 --l_r 1e-4 --epoches 100
```
```
python Run_KGE.py --model ConvKB --dataset WN18RR --n_filter 32 --dropout 0.4 --l2 5e-3 --epoches 20 --earlystop 3 --add_rgcn True
```
