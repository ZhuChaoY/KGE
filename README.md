# KGE
Knowledge Graph Embedding Models

# Files
### dataset/
FB15k (larger than 25MB, first unzip the file)  
FB15k-237  
WN18  
WN18RR  

**KGE.py** : Class of processing and tool functions for Knowledge Graph Embedding.  
**Models.py** : TransE, TransH, TransR, TransD, ConvKB structure.  
**Run_KGE.py** : Train KGE model.  

# Operating Instructions
Run Run_KGE.py to train TransE, TransH, TransR, TransD, ConvKB.  
**TransX**:   
```
$ python Run_KGE.py --model TransE --dataset WN18 --dim 128 --margin 1.0 --l_r 1e-3 --batch_size 4800 --epoches 500 --do_train True --save_model False --do_predict True
```
**ConvKB**:  
```
$ python Run_KGE.py --model ConvKB -dataset WN18 --dim 128 --n_filter 8 --dropout 0.1 --l_r 1e-3 --batch_size 4800 --epoches 500 --do_train True --save_model False --do_predict True
