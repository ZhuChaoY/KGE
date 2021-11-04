# KGE
A Framework of Knowledge Graph Embedding Models (Including TransE, TransH, TransR, TransD, ConvKB now, keep expanding)

##Files
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
