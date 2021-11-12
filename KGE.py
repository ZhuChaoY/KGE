import re
import time
import json
import random
import collections
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from os import makedirs
from os.path import exists


class KGE():
    """
    A Framework of R-GCN enhanced Knowledge Graph Embedding Models.
    (1) Pre-train KGE models by traditional process.
    (2) Serve a single layer of R-GCN as the encoder (The pre trained entity
        embeddings are the input feature of R-GCN), and a KGE model as the
        docoder, fine-tuning the pre-trained KGE models by few epoches.
    """
    
    def __init__(self, args):
        """
        (1) Initialize KGE with args dict.
        (2) Named data dir and out dir.
        (3) Load entity, relation and triple.
        (4) Load common model structure.
        """
        
        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
                        
        self.data_dir = 'dataset/{}/'.format(self.dataset)
        self.out_dir = '{}{}/{}_{}'.format(self.data_dir, self.model,
                        self.dim, self.margin if self.model != 'ConvKB'
                        else self.n_filter)
        if self.add_rgcn:
            self.out_dir += ' (add R-GCN)'
        if not exists(self.out_dir):
            makedirs(self.out_dir)
        
        print('\n\n' + '==' * 4 + ' < {} > && < {} > {}'.format(self.model,
             self.dataset, '(add R-GCN) ' if self.add_rgcn else '') + '==' * 4)         
        self.em_data()
        self.common_structure()
    
    
    def em_data(self):
        """
        (1) Get entity mapping dict (E_dict).
        (2) Get relation mapping dict (R_dict).
        (3) Get train, dev and test dataset for embedding.
        (4) Get replace_h_prob dict and triple pool for negative 
            sample's generation.
        """
        
        self.E_dict = {}
        for line in open(self.data_dir + 'entities.txt', 'r+'):
            line = line.strip().split('\t')
            self.E_dict[line[1]] = int(line[0])
        self.n_E = len(self.E_dict) 
        print('    #Entity   : {}'.format(self.n_E))
        self.R_dict = {}
        for line in open(self.data_dir + 'relations.txt', 'r+'):
            line = line.strip().split('\t')
            self.R_dict[line[1]] = int(line[0])
        self.n_R = len(self.R_dict)
        print('    #Relation : {}'.format(self.n_R))
        
        for key in ['train', 'dev', 'test']:
            T = []
            for line in open(self.data_dir + key + '.txt', 'r+'):
                h, r, t = line.strip().split('\t')
                T.append([self.E_dict[h], self.R_dict[r], self.E_dict[t]])
            T = np.array(T)
            n_T = len(T)
            exec('self.' + key + ' = T')
            exec('self.n_' + key + ' = n_T')
            print('    #{:5} : {:6} ({:>5} E + {:>4} R)'.format( \
                  key.title(), n_T, len(set(T[:, 0]) | set(T[:, 2])),
                  len(set(T[:, 1]))))
                                             
        rpc_h_prob = {} #Bernoulli Trick
        for r in range(self.n_R):
            idx = np.where(self.train[:, 1] == r)[0]
            t_per_h = len(idx) / len(set(self.train[idx, 0]))
            h_per_t = len(idx) / len(set(self.train[idx, 2]))
            rpc_h_prob[r] = t_per_h / (t_per_h + h_per_t)
        self.rpc_h = lambda r : np.random.binomial(1, rpc_h_prob[r])
        
        self.pool = {tuple(x) for x in self.train.tolist() +
                     self.dev.tolist() + self.test.tolist()}
        
    
    def common_structure(self):
        """The common structure of R-GCN KGE model."""
        
        print('\n    *Embedding Dim   : {}'.format(self.dim))
        if self.model != 'ConvKB':
            print('    *Margin          : {}'.format(self.margin))
        else:
            print('    *N_filter        : {}'.format(self.n_filter)) 
        print('    *Dropout         : {}'.format(self.dropout))
        print('    *l2 Rate         : {}'.format(self.l2))
        print('    *Learning_Rate   : {}'.format(self.l_r))
        print('    *Batch_Size      : {}'.format(self.batch_size))
        print('    *Max Epoches     : {}'.format(self.epoches))
        print('    *Earlystop Steps : {}'.format(self.earlystop))
        
        tf.reset_default_graph()
        self.T_pos = tf.placeholder(tf.int32, [None, 3])
        self.T_neg = tf.placeholder(tf.int32, [None, 2])
        self.keep = tf.placeholder(tf.float32)
        self.K = np.sqrt(6.0 / self.dim)
        
        if self.add_rgcn:        
            A = self.get_A()
            self.supports = [tf.sparse_placeholder(tf.float32)
                             for _ in range(self.n_R)]
            self.feed_dict = {self.supports[r]: A[r] for r in range(self.n_R)}
        
            with tf.variable_scope('R-GCN'): 
                self.input = tf.get_variable('input_feature', [self.n_E,
                             self.dim], trainable = False)
                E_table = self.rgcn_layer()
                
        with tf.variable_scope('structure'): 
            if not self.add_rgcn:
                E_table = tf.get_variable('entity_table', initializer = \
                      tf.random_uniform([self.n_E, self.dim], -self.K, self.K))
            E_table = tf.nn.l2_normalize(E_table, 1)    
            R_table = tf.get_variable('relation_table', initializer = \
                      tf.random_uniform([self.n_R, self.dim], -self.K, self.K))
            R_table = tf.nn.l2_normalize(R_table, 1)

            if self.model == 'TransR':
                E_table = tf.reshape(E_table, [-1, 1, self.dim])
                R_table = tf.reshape(R_table, [-1, 1, self.dim])
            elif self.model == 'ConvKB':
                E_table = tf.reshape(E_table, [-1, self.dim, 1, 1])
                R_table = tf.reshape(R_table, [-1, self.dim, 1, 1])
            h_pos = tf.gather(E_table, self.T_pos[:, 0])
            t_pos = tf.gather(E_table, self.T_pos[:, -1])
            h_neg = tf.gather(E_table, self.T_neg[:, 0])
            t_neg = tf.gather(E_table, self.T_neg[:, -1])
            r = tf.gather(R_table, self.T_pos[:, 1])
            
            self.l2_s = [h_pos, t_pos, h_neg, t_neg, r]
            s_pos, s_neg = self.em_structure(h_pos, t_pos, h_neg, t_neg, r)

        with tf.variable_scope('score'): 
            self.score_pos, self.score_neg = self.cal_score(s_pos, s_neg)

        with tf.variable_scope('loss'): 
            if self.model != 'ConvKB':
                loss = tf.reduce_sum(tf.nn.relu(self.margin + \
                       self.score_pos - self.score_neg))
            else:
                loss = tf.reduce_sum(tf.nn.softplus(self.score_pos) + \
                                     tf.nn.softplus(- self.score_neg))
            l2_loss_s = tf.add_n([tf.nn.l2_loss(v) for v in self.l2_s])
            if not self.add_rgcn:
                self.loss = loss + self.l2 * l2_loss_s
            else:
                l2_loss_r = tf.add_n([tf.nn.l2_loss(v) for v in self.l2_r])
                self.loss = loss + self.l2 * (l2_loss_r + l2_loss_s / 5)
            self.train_op = tf.train.AdamOptimizer(self.l_r). \
                            minimize(self.loss)


    def get_A(self):
        """Get adjacency matrix for each relations, normalized to row sum 1."""
        
        A = []
        for r in range(self.n_R):
            edges = self.train[self.train[:, 1] == r]
            edges = np.delete(edges, 1, 1)
            row, col = np.transpose(edges)
            n = edges.shape[0]
            data = np.array([1 / n for _ in range(n)])
            a = sp.coo_matrix((data, (row, col)), shape = (self.n_E, self.n_E))
            A.append((np.vstack((a.row, a.col)).transpose(), a.data, a.shape))
        return A

    
    def rgcn_layer(self):
        """
        A layer of R-GCN.
        Set n_B == 32.
        If n_R <= n_B: don't apply basis decompasition.
        """
        
        K = np.sqrt(3.0 / self.dim)
        
        s_w = tf.get_variable('self_weight', initializer = \
              tf.random_uniform([self.dim, self.dim], -K, K))
        out = tf.nn.dropout(tf.matmul(self.input, s_w), 0.5 * self.keep + 0.5)
        self.l2_r = [s_w]
            
        n_B = 32
        if self.n_R <= n_B:
            r_w = tf.get_variable('relation_weight', initializer = \
                  tf.random_uniform([self.n_R, self.dim, self.dim], -K, K))
            self.l2_r.append(r_w)
        else:
            r_c = tf.get_variable('relation_coefficient', initializer = \
                  tf.random_uniform([self.n_R, n_B], -K, K))
            r_b = tf.get_variable('relation_basis', initializer = \
                  tf.random_uniform([n_B, self.dim, self.dim], -K, K))
            r_w = tf.reshape(tf.matmul(r_c, tf.reshape(r_b, [-1, self.dim * \
                  self.dim])), [-1, self.dim, self.dim])
            self.l2_r.append(r_b)
        
        for r in range(self.n_R):
            out = tf.nn.dropout(tf.matmul(tf.sparse_tensor_dense_matmul( \
                  self.supports[r], self.input), r_w[r]), self.keep) + out

        return out
    

    def em_train(self, sess):  
        """
        (1) Initialize and display variables and shapes.
        (1) Training and Evalution process of embedding.
        (2) Calculate loss for train and dev dataset, check whether reach
            the earlystop steps.
            
        Args:
            sess: tf.Session
        """

        eps = self.epoches
        step = eps if self.add_rgcn else 25
        bps = list(range(eps // step - 1, eps, eps // step))
        print('    EPOCH Trian-LOSS Dev-LOSS  time   Time')  
            
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(eps):  
            train_batches = self.get_batches('train')
            train_Loss = 0.0
            for T_pos, T_neg in train_batches:     
                feed_dict = {self.T_pos: T_pos, self.T_neg: T_neg,
                             self.keep: 1.0 - self.dropout}
                if self.add_rgcn:
                    feed_dict.update(self.feed_dict)
                loss, _ = sess.run([self.loss, self.train_op], feed_dict)
                train_Loss += loss         
            train_Loss = round(train_Loss / self.n_train / 2, 4)
            
            if ep in bps:
                dev_batches = self.get_batches('dev')
                dev_Loss = 0.0
                for T_pos, T_neg in dev_batches:     
                    feed_dict = {self.T_pos: T_pos, self.T_neg: T_neg,
                                 self.keep: 1.0}
                    if self.add_rgcn:
                        feed_dict.update(self.feed_dict)
                    loss = sess.run(self.loss, feed_dict)
                    dev_Loss += loss   
                dev_Loss = round(dev_Loss / self.n_dev, 4)
                
                _t = time.time()
                print('    {:^5} {:^10.4f} {:^8.4f} {:^6.2f} {:^6.2f}'. \
                      format(ep + 1, train_Loss, dev_Loss, (_t - t1) / 60,
                             (_t - t0) / 60))
                t1 = _t
                
                if ep == bps[0] or dev_Loss < KPI[-1]:
                    tf.train.Saver().save(sess, self.out_dir + '/model.ckpt')
                    if len(temp_kpi) > 0:
                        KPI.extend(temp_kpi)
                        temp_kpi = []
                    KPI.append(dev_Loss)
                else:
                    if len(temp_kpi) == self.earlystop:
                        break
                    else:
                        temp_kpi.append(dev_Loss)
        
        best_ep = bps[len(KPI) - 1] + 1
        if best_ep != eps:
            print('\n    Early stop at epoch of {} !'.format(best_ep))
    
        result = {'args': self.args, 'dev-loss': KPI,
                  'best-epoch': best_ep}            
        with open(self.out_dir + '/result.json', 'w') as file: 
            json.dump(result, file) 


    def initialize_variables(self, mode):
        """
        Initialize and display variables and shapes.
        
        Args:
            mode: 'train' or 'predict'
        """
        
        tvs = collections.OrderedDict()
        for v in tf.global_variables():
            name = re.match('^(.*):\\d+$', v.name).group(1)
            shape = v.shape.as_list()
            if 'Adam' not in name and 'beta' not in name:
                tvs[name] = shape
                
        if mode == 'train':
            if not self.add_rgcn:
                if self.model == 'ConvKB':                
                    p = '{}TransE/{}_{}'.format(self.data_dir, self.dim,
                        '1.0' if 'FB' in self.dataset else '1.5')
                else:
                    p = None
            else:
                p = self.out_dir[: -12]
        elif mode == 'predict':
            p = self.out_dir
        
        if p:
            p += '/model.ckpt'
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
            if self.add_rgcn and mode == 'train':
                ivs['structure/entity_table'] = 'R-GCN/input_feature'
            tf.train.init_from_checkpoint(p, ivs)
        else:
            ivs = {}
                                
        if mode == 'train' or (mode == 'predict' and not self.do_train):
            for v, shape in tvs.items():
                print('    {}{} : {}'.format('*' if v in ivs or 'feature' in v
                      else '-', v, shape))
            print()


    def get_batches(self, key):
        """
        Get postive batch triple (T_pos) for training.
        
        Args:
            key: 'train' or 'dev'
        """
        
        bs = self.batch_size
        data = eval('self.' + key + '.copy()')
        n = len(data)
        random.shuffle(data)                    
        n_batch = n // bs
        T_poss = [data[i * bs: (i + 1) * bs] for i in range(n_batch)]
        if n % bs != 0:
            T_poss.append(data[n_batch * bs: ])
            
        if key == 'train':
            return ((np.vstack([T_pos, T_pos]), np.vstack([self.get_T_neg( \
                     T_pos), self.get_T_neg(T_pos)])) for T_pos in T_poss) 
        elif key == 'dev':
            return ((T_pos, self.get_T_neg(T_pos)) for T_pos in T_poss) 
    
    
    def get_T_neg(self, T_pos):
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
                    T_neg.append((new_T[0], new_T[2]))
                    break
        return np.array(T_neg)


    def em_predict(self, sess):
        """
        Predict for test dataset.
        
        Args:
            sess: tf.Session
        """
             
        t0 = time.time()
        print('     MR    MRR   @01   @03   @10   TIME\n   ', end = '')
        out = self.link_prediction(sess, self.test)
        print(' {:^6.2f}'.format((time.time() - t0) / 60))
        
        with open(self.out_dir + '/result.json') as file: 
            result = json.load(file) 
        
        result.update(out)
        
        with open(self.out_dir + '/result.json', 'w') as file: 
            json.dump(result, file) 
    
    
    def link_prediction(self, sess, T_pos):   
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
            feed_dict = {self.T_pos: rpc_h, self.keep: 1.0}
            if self.add_rgcn:
                feed_dict.update(self.feed_dict)
            score_h = sess.run(self.score_pos, feed_dict)
            
            rpc_t = np.array([T for i in range(self.n_E)])
            rpc_t[:, 2] = range(self.n_E)
            feed_dict = {self.T_pos: rpc_t, self.keep: 1.0}
            if self.add_rgcn:
                feed_dict.update(self.feed_dict)
            score_t = sess.run(self.score_pos, feed_dict)

            rank.extend([self.cal_ranks(score_h, T, 0), 
                         self.cal_ranks(score_t, T, 2)])    
        
        MR = round(np.mean(rank), 1)
        MRR = round(np.mean([1 / x for x in rank]), 3)
        top1 = round(np.mean(np.array(rank) == 1), 3)
        top3 = round(np.mean(np.array(rank) <= 3), 3)
        top10 = round(np.mean(np.array(rank) <= 10), 3)
        
        print('{:>6.1f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f}'. \
              format(MR, MRR, top1, top3, top10), end = '')
        
        return {'MR': MR, 'MRR': MRR, '@1': top1, '@3': top3, '@10': top10}
    
    
    def cal_ranks(self, score, T, idx):
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
    
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        if self.do_train:
            print('\n>>  Training Process.')
            self.initialize_variables('train')        
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()   
                self.em_train(sess)
                
        if self.do_predict:
            print('\n>>  Predict Process.')
            self.initialize_variables('predict')   
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()  
                self.em_predict(sess)
                