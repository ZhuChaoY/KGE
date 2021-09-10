import re
import time
import random
import numpy as np
import tensorflow.compat.v1 as tf
from os import makedirs
from os.path import exists


class KGE():
    """
    A class of processing and tool functions for Knowledge Graph Embedding.
    """
    
    def __init__(self, args):
        """
        (1) Initialize KGE with args dict.
        Args:
            'model'      : which KGE model.
            'dataset'    : knowledge graph embedding model for which dataset.
            'dim'        : embedding dim.
            'margin'     : margin hyperparameter.
            'n_filter'   : number of filters.
            'dropout'    : dropout rate.
            'l_r'        : learning rate.
            'batch_size' : batch size for training.
            'epoches'    : training epoches.
            'do_train'   : whether to train the model.
            'do_predict' : whether to predict for test dataset.
        (2) Named model dir and out dir.
        (3) Load entity, relation and triple.
        (4) Load model structure.
        (5) Initialize embedding trainable variables.
        """
        
        for key, value in dict(args._get_kwargs()).items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
                        
        self.dir = 'dataset/' + self.dataset + '/'
        self.out_dir = self.dir + self.model + '/'
        if not exists(self.out_dir):
            makedirs(self.out_dir)
        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)
        
        print('\n' + '==' * 4 + ' < {} > && < {} >'.format(self.model,
              self.dataset) + '==' * 4)        
        self._em_data()
        self._common_structure()
    
              
    def _em_data(self):
        """
        (1) Get entity mapping dict (E_dict).
        (2) Get relation mapping dict (R_dict), and the index for activate 
            and inhibit.
        (3) Get train, valid and test dataset for embedding.
        (4) Get replace_h_prob dict and triple pool for negative 
            sample's generation.
        """
        
        self.E_dict = {}
        for line in open(self.dir + 'entities.txt', 'r+'):
            line = line.strip().split('\t')
            self.E_dict[line[1]] = int(line[0])
        self.n_E = len(self.E_dict) 
        print('    #Entity   : {}'.format(self.n_E))
        self.R_dict = {}
        for line in open(self.dir + 'relations.txt', 'r+'):
            line = line.strip().split('\t')
            self.R_dict[line[1]] = int(line[0])
        self.n_R = len(self.R_dict)
        print('    #Relation : {}'.format(self.n_R))
        
        for key in ['train', 'valid', 'test']:
            T = []
            for line in open(self.dir + key + '.txt', 'r+'):
                h, r, t = line.strip().split('\t')
                T.append([self.E_dict[h], self.R_dict[r], self.E_dict[t]])
            T = np.array(T)
            n_T = len(T)
            exec('self.' + key + ' = T')
            exec('self.n_' + key + ' = n_T')
            print('    #{:5} : {:6} ({:>5} E + {:>4} R)'.format( \
                  key.title(), n_T, len(set(T[:, 0]) | set(T[:, 2])),
                  len(set(T[:, 1]))))
                            
        rpc_h_prob = {} #'Bernoulli Trick'
        for r in range(self.n_R):
            idx = np.where(self.train[:, 1] == r)[0]
            t_per_h = len(idx) / len(set(self.train[idx, 0]))
            h_per_t = len(idx) / len(set(self.train[idx, 2]))
            rpc_h_prob[r] = t_per_h / (t_per_h + h_per_t)
        self.rpc_h = lambda r : np.random.binomial(1, rpc_h_prob[r])
        
        self.pool = {tuple(x) for x in self.train.tolist() +
                     self.valid.tolist() + self.test.tolist()}
    
    
    def _common_structure(self):
        """The common structure of KGE model."""
        
        print('\n    *Dim           : {}'.format(self.dim))
        if self.model != 'ConvKB':
            if self.margin:
                print('    *Margin        : {}'.format(self.margin))
            else:
                raise 'Lack margin value!'
        else:
            if self.n_filter:
                print('    *N_filter      : {}'.format(self.n_filter))
            else:
                raise 'Lack n_filter value!'
        print('    *Dropout       : {}'.format(self.dropout))
        print('    *Learning_Rate : {}'.format(self.l_r))
        print('    *Batch_Size    : {}'.format(self.batch_size))
        print('    *Epoches       : {}'.format(self.epoches))
        
        tf.reset_default_graph()
        self.T_pos = tf.placeholder(tf.int32, [None, 3])
        self.T_neg = tf.placeholder(tf.int32, [None, 3])
        self.keep = tf.placeholder(tf.float32)
        
        with tf.variable_scope('structure'): #(B, D)
            E_table = tf.nn.l2_normalize(tf.get_variable('entity_table',
                      [self.n_E, self.dim], initializer = self.initializer), 1)
            R_table = tf.nn.l2_normalize(tf.get_variable('relation_table',
                      [self.n_R, self.dim], initializer = self.initializer), 1)
            
            if self.model == 'TransR':
                E_table = tf.reshape(E_table, [-1, 1, self.dim])
                R_table = tf.reshape(R_table, [-1, 1, self.dim])
            elif self.model == 'ConvKB':
                E_table = tf.reshape(E_table, [-1, self.dim, 1, 1])
                R_table = tf.reshape(R_table, [-1, self.dim, 1, 1])
                
            h_pos = tf.gather(E_table, self.T_pos[:, 0])
            r_pos = tf.gather(R_table, self.T_pos[:, 1])
            t_pos = tf.gather(E_table, self.T_pos[:, 2])
            h_neg = tf.gather(E_table, self.T_neg[:, 0])
            r_neg = tf.gather(R_table, self.T_neg[:, 1])
            t_neg = tf.gather(E_table, self.T_neg[:, 2])
            
            s_pos, s_neg = self._em_structure(h_pos, r_pos, t_pos,
                                              h_neg, r_neg, t_neg)
            
        with tf.variable_scope('score'): #(B, 1)
            self.score_pos, self.score_neg = self._cal_score(s_pos, s_neg)
            
        with tf.variable_scope('loss'): #(1)
            self.loss = self._cal_loss()
            self.train_op = tf.train.AdamOptimizer(self.l_r). \
                            minimize(self.loss)
            
        self._em_init() 
    
    
    
    def _em_init(self):
        """Initialize embedding trainable variables."""
        
        shape = {re.match('^(.*):\\d+$', v.name).group(1):
                  v.shape.as_list() for v in tf.trainable_variables()}
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
                for v in tf.trainable_variables()]
                       
        if not self.do_train:
            p = self.out_dir + 'model.ckpt'
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                    if v[0] in tvs}
            tf.train.init_from_checkpoint(p, ivs)
        else:
            ivs = {}
            
        print('\n>>  {} of {} trainable variables initialized.'. \
              format(len(ivs), len(tvs)))  
        for v in tvs:
            print('    {}{} : {}'.format('*' if v in ivs else '-', v, 
                                          shape[v]))
            
    
    def _em_train(self, sess):  
        """
        (1) Training and Evalution process of embedding.
        (2) Evaluate for dev dataset totally 10 breakpoints during training,
            evaluate for train dataset lastly.
        
        Args:
            sess: tf.Session
        """

        bs, n_train, eps = self.batch_size, self.n_train, self.epoches
        n_batch = n_train // bs
        bps = list(range(eps // 10 - 1, eps, eps // 10))
        print('\n>>  Training Process. ({} EPOCHES) '.format(eps))

        print('    EPOCH Trian-LOSS Valid:  MR    MRR   @01   @03   @10 '
              '  time   TIME')  
            
        t0 = t1 = time.time()
        for ep in range(eps):
            sample = random.sample(range(n_train), n_train)
            idxes = [sample[i * bs: (i + 1) * bs] for i in range(n_batch)]
        
            Loss = 0.0
            for idx in idxes:     
                T_pos = self.train[idx]
                T_neg = self._get_T_neg(T_pos)
                feed_dict = {self.T_pos: T_pos, self.T_neg: T_neg,
                             self.keep: 1.0 - self.dropout}
                loss, _ = sess.run([self.loss, self.train_op], feed_dict)
                Loss += loss         
      
            if ep in bps:
                print('    {:^5} {:^10.4f}       '. \
                      format(ep + 1, Loss / n_batch / bs), end = '')
                self._link_prediction(sess, self.valid[:100])                     
                _t = time.time()
                print(' {:^6.2f} {:^6.2f}'. \
                      format((_t - t1) / 60, (_t - t0) / 60))
                t1 = _t
        
        if self.do_predict:
            tf.train.Saver().save(sess, self.out_dir + 'model.ckpt')
    
    
    def _get_T_neg(self, T_pos):
        """
        (1) Get negative triple (T_neg) for training.
        (2) Replace head or tail depends on replace_h_prob.
        
        Args:
            T_pos: positive triples
        """
        
        T_neg = []
        for h, r, ta in T_pos.tolist():
            while True:    
                new_e = random.choice(range(self.n_E))
                new_T = (new_e, r, ta) if self.rpc_h(r) else (h, r, new_e)
                if new_T not in self.pool:
                    T_neg.append(new_T)
                    break
        return np.array(T_neg)
    
    
    def _link_prediction(self, sess, T_pos):   
        """
        Linking Prediction of knowledge graph embedding.
        Return entity MR, MRR, @1, @3, @10
        
        Args:
            sess: tf.Session
            T_pos: positive triple to predict
        """
        
        rank = []
        for T in T_pos.tolist():      
            rpc_h = np.array([T for i in range(self.n_E)])
            rpc_h[:, 0] = range(self.n_E)
            score_h = sess.run(self.score_pos, {self.T_pos: rpc_h, 
                                                self.keep: 1.0})
            
            rpc_t = np.array([T for i in range(self.n_E)])
            rpc_t[:, 2] = range(self.n_E)
            score_t = sess.run(self.score_pos, {self.T_pos: rpc_t,
                                                self.keep: 1.0})

            rank.extend([self._cal_ranks(score_h, T, 0), 
                         self._cal_ranks(score_t, T, 2)])    
        
        MR = round(np.mean(rank), 1)
        MRR = round(np.mean([1 / x for x in rank]), 3)
        top1 = round(np.mean(np.array(rank) == 1), 3)
        top3 = round(np.mean(np.array(rank) <= 3), 3)
        top10 = round(np.mean(np.array(rank) <= 10), 3)
        
        print('{:>6.1f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f}'. \
              format(MR, MRR, top1, top3, top10), end = '')
    
    
    def _cal_ranks(self, score, T, idx):
        """
        Cal link prediction rank for a single triple.
        
        Args:
            score: replace an entity (a relation) by all the entity of an real
            triple, shape of [n_E, 3]
            T: raw triple
            idx: the replace place of the triple
        """
        
        rank = np.argsort(score)
        out = 1 
        for x in rank:
            if x == T[idx]:
                break
            else:
                new_T = T.copy()
                new_T[idx] = x
                if tuple(new_T) not in self.pool:
                    out += 1
        return out


    def _em_predict(self, sess):
        """
        Predict for test dataset.
        
        Args:
            sess: tf.Session
        """
                
        print('\n>>  Test Link Prediction Result.')
        t0 = time.time()
        print('    MR    MRR   @01   @03   @10   TIME\n   ', end = '')
        self._link_prediction(sess, self.test)
        print(' {:^6.2f}'.format((time.time() - t0) / 60))
        
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()   
            if self.do_train:
                self._em_train(sess)
            if self.do_predict:
                self._em_predict(sess)
                
            