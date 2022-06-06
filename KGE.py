import re
import time
import json
import pickle
import random
import collections
import numpy as np
import tensorflow as tf
from os import makedirs
from os.path import exists


class KGE():
    """A Framework of Knowledge Graph Embedding Models."""
    
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
        self.out_dir = '{}{}/{}_{}/'.format(self.data_dir, self.model,
                        self.dim, self.margin if self.model != 'ConvKB'
                        else self.n_filter)
        if not exists(self.out_dir):
            makedirs(self.out_dir)
        
        print('\n\n' + '==' * 4 + ' < {} > && < {} > '.format(self.model,
             self.dataset) + '==' * 4)         
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
        """The common structure of KGE model."""
        
        print('\n    *Embedding Dim   : {}'.format(self.dim))
        if self.model != 'ConvKB':
            print('    *Margin          : {}'.format(self.margin))
        else:
            print('    *N_filter        : {}'.format(self.n_filter)) 
        print('    *l2 Rate         : {}'.format(self.l2))
        print('    *Learning_Rate   : {}'.format(self.l_r))
        print('    *Batch_Size      : {}'.format(self.batch_size))
        print('    *Max Epoches     : {}'.format(self.epoches))
        print('    *Earlystop Steps : {}'.format(self.earlystop))
        
        tf.reset_default_graph()
        self.T_pos = tf.placeholder(tf.int32, [None, 3])
        self.T_neg = tf.placeholder(tf.int32, [None, 2])
        self.K = np.sqrt(6.0 / self.dim)
        
        with tf.variable_scope('structure'): 
            self.E_table = tf.get_variable('entity_table', initializer = \
                      tf.random_uniform([self.n_E, self.dim], -self.K, self.K))
            self.E_table = tf.nn.l2_normalize(self.E_table, 1)    
            if self.model != 'RotatE':
                R_table = tf.get_variable('relation_table', initializer = \
                      tf.random_uniform([self.n_R, self.dim], -self.K, self.K))
                R_table = tf.nn.l2_normalize(R_table, 1)
            else:
                pi = 3.1415926
                R_table = tf.get_variable('relation_table', initializer = \
                          tf.random_uniform([self.n_R, self.dim], -pi, pi))

            if self.model == 'ConvKB':
                self.E_table = tf.reshape(self.E_table, [-1, self.dim, 1, 1])
                R_table = tf.reshape(R_table, [-1, self.dim, 1, 1])
            h_pos = tf.gather(self.E_table, self.T_pos[:, 0])
            t_pos = tf.gather(self.E_table, self.T_pos[:, -1])
            h_neg = tf.gather(self.E_table, self.T_neg[:, 0])
            t_neg = tf.gather(self.E_table, self.T_neg[:, -1])
            r = tf.gather(R_table, self.T_pos[:, 1])
            
            self.l2_kge = [h_pos, t_pos, r, h_neg, t_neg]
            self.kge_variables()
            s_pos = self.em_structure(h_pos, r, t_pos, 'pos')
            self.score_pos = self.cal_score(s_pos)
            s_neg = self.em_structure(h_neg, r, t_neg, 'neg')
            score_neg = self.cal_score(s_neg)

        with tf.variable_scope('loss'): 
            if self.model != 'ConvKB':
                loss = tf.reduce_sum(tf.nn.relu(self.margin + \
                       self.score_pos - score_neg))
            else:
                loss = tf.reduce_sum(tf.nn.softplus(self.score_pos) + \
                                     tf.nn.softplus(- score_neg))
            loss_kge = tf.add_n([tf.nn.l2_loss(v) for v in self.l2_kge])
            loss = loss + self.l2 * loss_kge
            self.train_op = tf.train.AdamOptimizer(self.l_r).minimize(loss)
                            
        with tf.variable_scope('link_prediction'): 
            self.lp_h, self.lp_t = self.cal_lp_score(h_pos, r, t_pos)


    def em_train(self, sess):  
        """
        (1) Initialize and display variables and shapes.
        (2) Training and Evalution process of embedding.
        (3) Calculate result of dev dataset, check whether reach the earlystop.
            
        Args:
            sess: tf.Session
        """

        eps = self.epoches
        bps = list(range(eps // 50 - 1, eps, eps // 50))
        print('    EPOCH   MR    MRR   @01   @03   @10   time   Time  (Dev)')  
            
        result = {'args': self.args}
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(eps):  
            T_poss = self.get_batches('train')
            for T_pos in T_poss:
                T_neg = self.get_T_neg(T_pos)
                feed_dict = {self.T_pos: T_pos, self.T_neg: T_neg}
                _ = sess.run(self.train_op, feed_dict)     
            
            if ep in bps:
                print('    {:^5} '.format(ep + 1), end = '')
                lp_out = self.link_prediction(sess, 'dev')
                kpi = lp_out['@10']
                _t = time.time()
                print(' {:^6.2f} {:^6.2f}'.format((_t - t1) / 60, 
                                                  (_t - t0) / 60), end = '')
                t1 = _t
            
                if ep == bps[0] or kpi > KPI[-1]:
                    print(' *')
                    if len(temp_kpi) > 0:
                        KPI.extend(temp_kpi)
                        temp_kpi = []
                    KPI.append(kpi)
                    tf.train.Saver().save(sess, self.out_dir + 'model.ckpt')
                    best_ep = bps[len(KPI) - 1] + 1                
                    result['dev-top10'] = KPI
                    result['best-epoch'] = best_ep            
                    with open(self.out_dir + 'result.json', 'w') as file: 
                        json.dump(result, file) 
                    
                else:
                    print('')
                    if len(temp_kpi) == self.earlystop:
                        break
                    else:
                        temp_kpi.append(kpi)
        
        if best_ep != eps:
            print('\n    Early stop at epoch of {} !'.format(best_ep))


    def get_batches(self, key):
        """
        Generate batch data by batch size.
        
        args:
            key: 'train', 'dev' or 'test'
        """
        
        bs = self.batch_size
        data = eval('self.' + key + '.copy()')
        if key == 'train':
            np.random.shuffle(data)
        else:
            bs = 50 if self.model != 'ConvKB' else 1
        n = len(data)
        n_batch = n // bs
        T_poss = [data[i * bs: (i + 1) * bs] for i in range(n_batch)]
        if n % bs != 0:
            T_poss.append(data[n_batch * bs: ])
        
        return T_poss 
    
    
    def get_T_neg(self, T_pos):
        """
        (1) Get negative triple (T_neg) for training.
        (2) Replace head or tail depends on replace_h_prob.
        
        Args:
            T_pos: positive triples
        """
        
        T_negs = []
        for h, r, ta in T_pos.tolist():
            while True:    
                new_e = random.choice(range(self.n_E))
                new_T = (new_e, r, ta) if self.rpc_h(r) else (h, r, new_e)
                if new_T not in self.pool:
                    T_negs.append((new_T[0], new_T[2]))
                    break
        return np.array(T_negs)


    def em_predict(self, sess):
        """
        Predict for test dataset.
        
        Args:
            sess: tf.Session
        """
             
        t0 = time.time()
        print('     MR    MRR   @01   @03   @10   TIME  (Test)\n   ', end = '')
        lp_out = self.link_prediction(sess, 'test')
        print(' {:^6.2f}'.format((time.time() - t0) / 60))
        
        with open(self.out_dir + 'result.json') as file: 
            result = json.load(file) 
        
        result.update(lp_out)
        
        with open(self.out_dir + 'result.json', 'w') as file: 
            json.dump(result, file) 
    
    
    def link_prediction(self, sess, key):   
        """
        Linking Prediction of knowledge graph embedding.
        Return entity MR, MRR, @1, @3, @10
        
        Args:
            sess: tf.Session
            key: 'dev' or 'test'
        """
        
        T_poss = self.get_batches(key)
        rank = []
        for T_pos in T_poss:
            feed_dict = {self.T_pos: T_pos}
            lp_h, lp_t = sess.run([self.lp_h, self.lp_t], feed_dict)
            for i in range(len(T_pos)):
                rank.extend([self.cal_ranks(lp_h[i], list(T_pos[i]), 0), 
                             self.cal_ranks(lp_t[i], list(T_pos[i]), 2)]) 
            
        MR = round(np.mean(rank), 1)
        MRR = round(np.mean([1 / x for x in rank]), 3)
        top1 = round(np.mean(np.array(rank) == 1), 3)
        top3 = round(np.mean(np.array(rank) <= 3), 3)
        top10 = round(np.mean(np.array(rank) <= 10), 5)
        
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
    
    
    def initialize_variables(self, mode):
        """
        Initialize and display variables and shapes.
        
        Args:
            mode: 'train' or 'predict'
        """
        
        tvs = collections.OrderedDict()
        for v in tf.trainable_variables():
            name = re.match('^(.*):\\d+$', v.name).group(1)
            shape = v.shape.as_list()
            tvs[name] = shape
                
        if mode == 'train':
            if self.model == 'ConvKB':    
                if 'FB' in self.dataset:
                    margin = 1.0
                elif 'WN' in self.dataset:
                    margin = 4.0
                elif self.dataset == 'NELL-995':
                    margin = 5.0
                elif self.dataset in ['Kinship', 'UMLS']:
                    margin = 0.1
                p = '{}TransE/{}_{}/'.format(self.data_dir, self.dim, margin)
            else:
                p = None
        else:
            p = self.out_dir
        
        if p:
            p += 'model.ckpt'
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
            tf.train.init_from_checkpoint(p, ivs)
        else:
            ivs = {}
                                
        if mode == 'train' or (mode != 'train' and not self.do_train):
            for v, shape in tvs.items():
                print('    {}{} : {}'.format('*' if v in ivs else '-', v,
                                             shape))
            print()
    
    
    def em_evaluate(self, sess):
        """
        Cal scores for all possible triplets.
        
        Args:
            sess: tf.Session
        """

        t0 = time.time()
        print('>>  Cal scores for {} * {} * {} = {} pairs.'. \
              format(self.n_E, self.n_R, self.n_E,
                     self.n_E * self.n_R * self.n_E))
        S = np.zeros((self.n_E, self.n_E, self.n_R))
        for h in range(self.n_E):
            for t in range(self.n_E):
                if t == h:
                    S[h, t] = np.array([None for i in range(self.n_R)])
                else:
                    T = np.array([[h, r, t] for r in range(self.n_R)])
                    s = sess.run(self.score_pos, {self.T_pos: T})
                    for r in range(self.n_R):
                        if (h, r, t) in self.pool:
                            s[r] = None
                    S[h, t] = s
        with open(self.out_dir + 'score.data', 'wb') as file:
            pickle.dump(S, file) 
        print('    {:.2f} min.'.format((time.time() - t0) / 60)) 
    
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
                        
        for mode in ['train', 'predict', 'evaluate']:
            if eval('self.do_' + mode):
                print('\n>>  {} Process.'.format(mode.title()))
                self.initialize_variables(mode)        
                with tf.Session(config = config) as _:
                    tf.global_variables_initializer().run()   
                    exec('self.em_' + mode + '(_)')