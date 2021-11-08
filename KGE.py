import re
import time
import json
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
            'l_r'        : learning rate.
            'batch_size' : batch size for training.
            'epoches'    : training epoches.
            'do_train'   : whether to train the model.
            'do_predict' : whether to predict for test dataset.
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
                        
        self.data_dir = 'dataset/' + self.dataset + '/'
        self.out_dir = self.data_dir + self.model + '/' + str(self.dim) + '_'
        if self.model != 'ConvKB':
            self.out_dir += str(self.margin)
        else:
            self.out_dir += str(self.n_filter)
        if not exists(self.out_dir):
            makedirs(self.out_dir)
        self.K = np.sqrt(6.0 / self.dim)
        
        print('\n\n' + '==' * 4 + ' < {} > && < {} >'.format(self.model,
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
        
        print('\n    *Dim             : {}'.format(self.dim))
        if self.model != 'ConvKB':
            if self.margin:
                print('    *Margin          : {}'.format(self.margin))
            else:
                raise 'Lack margin value!'
        else:
            if self.n_filter:
                print('    *N_filter        : {}'.format(self.n_filter))
            else:
                raise 'Lack n_filter value!'
        print('    *l2 Rate         : {}'.format(self.l2))
        print('    *Learning_Rate   : {}'.format(self.l_r))
        print('    *Batch_Size      : {}'.format(self.batch_size))
        print('    *Epoches         : {}'.format(self.epoches))
        print('    *Earlystop Steps : {}'.format(self.earlystop))
        
        tf.reset_default_graph()
        self.T_pos = tf.placeholder(tf.int32, [None, 3])
        self.T_neg = tf.placeholder(tf.int32, [None, 2])
        
        with tf.variable_scope('structure'): #(B, D)
            E_table = tf.nn.l2_normalize(tf.get_variable('entity_table', initializer = \
                      tf.random_uniform([self.n_E, self.dim], -self.K, self.K)), 1)
            R_table = tf.nn.l2_normalize(tf.get_variable('relation_table', initializer = \
                      tf.random_uniform([self.n_R, self.dim], -self.K, self.K)), 1)
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
            
            self.l2_v = [h_pos, t_pos, h_neg, t_neg, r]
            s_pos, s_neg = self.em_structure(h_pos, t_pos, h_neg, t_neg, r)
            
        with tf.variable_scope('score'): #(B, 1)
            self.score_pos, self.score_neg = self.cal_score(s_pos, s_neg)
            
        with tf.variable_scope('loss'): #(1)
            if self.model != 'ConvKB':
                loss = tf.reduce_sum(tf.nn.relu(self.margin + \
                       self.score_pos - self.score_neg))
            else:
                loss = tf.reduce_sum(tf.nn.softplus(self.score_pos) + \
                                     tf.nn.softplus(- self.score_neg))
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.l2_v])
            self.loss = loss + self.l2 * l2_loss
            self.train_op = tf.train.AdamOptimizer(self.l_r). \
                            minimize(self.loss)
            
        self.show_variables()
        
            
    def show_variables(self):
        """Display all variables and shapes."""
        
        shape = {re.match('^(.*):\\d+$', v.name).group(1):
                 v.shape.as_list() for v in tf.trainable_variables()}
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables()]
        print('')                         
        for v in tvs:
            print('    -{} : {}'.format(v, shape[v]))
            
        if self.model == 'ConvKB' and self.do_train:
            p = self.data_dir + 'TransE/' + str(self.dim) + '_1.0/model.ckpt'
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
            tf.train.init_from_checkpoint(p, ivs)
            print('    Initialize {} variables from TransE.'.format(len(ivs))) 
                    
    
    def em_train(self, sess):  
        """
        (1) Training and Evalution process of embedding.
        (2) Calculate loss for dev dataset totally 25 breakpoints during
            training, check whether reach the earlystop steps.
            
        
        Args:
            sess: tf.Session
        """

        eps = self.epoches
        bps = list(range(eps // 25 - 1, eps, eps // 25))
        print('\n>>  Training Process. (Max to {} EPOCHES) '.format(eps))
        print('    EPOCH Trian-LOSS Dev-LOSS  time   Time')  
            
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(eps):        
            train_batches = self.get_batches('train')
            train_Loss = 0.0
            for T_pos, T_neg in train_batches:     
                feed_dict = {self.T_pos: T_pos, self.T_neg: T_neg}
                loss, _ = sess.run([self.loss, self.train_op], feed_dict)
                train_Loss += loss         
            train_Loss = round(train_Loss / self.n_train, 4)
      
            if ep in bps:
                dev_batches = self.get_batches('dev')
                dev_Loss = 0.0
                for T_pos, T_neg in dev_batches:     
                    feed_dict = {self.T_pos: T_pos, self.T_neg: T_neg}
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
            score_h = sess.run(self.score_pos, {self.T_pos: rpc_h})
            
            rpc_t = np.array([T for i in range(self.n_E)])
            rpc_t[:, 2] = range(self.n_E)
            score_t = sess.run(self.score_pos, {self.T_pos: rpc_t})

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


    def em_predict(self, sess):
        """
        Predict for test dataset.
        
        Args:
            sess: tf.Session
        """
                
        print('\n>>  Predict Process.')
        self.initialize_variables()
        t0 = time.time()
        print('     MR    MRR   @01   @03   @10   TIME\n   ', end = '')
        out = self.link_prediction(sess, self.test)
        print(' {:^6.2f}'.format((time.time() - t0) / 60))
        
        with open(self.out_dir + '/result.json') as file: 
            result = json.load(file) 
        
        result.update(out)
        
        with open(self.out_dir + '/result.json', 'w') as file: 
            json.dump(result, file) 
        
        
    def initialize_variables(self):
        """Initialize all variables."""
        
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables()]
                       
        p = self.out_dir + '/model.ckpt'
        ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
               if v[0] in tvs}
        tf.train.init_from_checkpoint(p, ivs)
        
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()   
            if self.do_train:
                self.em_train(sess)
            if self.do_predict:
                self.em_predict(sess)
                