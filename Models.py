from KGE import KGE
import numpy as np
import tensorflow.compat.v1 as tf


class TransE(KGE):
    """Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, args):
        super().__init__(args)
            
    
    def _em_structure(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        s_pos = h_pos + r_pos - t_pos
        s_neg = h_neg + r_neg - t_neg
        
        return s_pos, s_neg
    
    
    def _cal_score(self, s_pos, s_neg):
        score_pos = tf.reduce_sum(s_pos ** 2, 1)
        score_neg = tf.reduce_sum(s_neg ** 2, 1)
        
        return score_pos, score_neg
        
    
    def _cal_loss(self):
        loss = tf.reduce_sum(tf.nn.relu(self.margin + self.score_pos - \
                                        self.score_neg))
        
        return loss
    
            

class TransH(KGE):
    """Knowledge Graph Embedding by Translating on Hyperplanes."""
    
    def __init__(self, args):
        super().__init__(args)
            
        
    def _em_structure(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        _projector = lambda s, p: \
            s - tf.reduce_sum(p * s, 1, keepdims = True) * p
        
        P_table = tf.nn.l2_normalize(tf.get_variable('projection_table',
                  [self.n_R, self.dim], initializer = self.initializer), 1)
        p_pos = tf.gather(P_table, self.T_pos[:, 1])
        p_neg = tf.gather(P_table, self.T_neg[:, 1])
        
        h_pos = _projector(h_pos, p_pos)
        t_pos = _projector(t_pos, p_pos)
        h_neg = _projector(h_neg, p_neg)
        t_neg = _projector(t_neg, p_neg)
                    
        s_pos = h_pos + r_pos - t_pos
        s_neg = h_neg + r_neg - t_neg
        
        return s_pos, s_neg
    
    
    def _cal_score(self, s_pos, s_neg):
        score_pos = tf.reduce_sum(s_pos ** 2, 1)
        score_neg = tf.reduce_sum(s_neg ** 2, 1)
        
        return score_pos, score_neg
        
    
    def _cal_loss(self):
        loss = tf.reduce_sum(tf.nn.relu(self.margin + self.score_pos - \
                                        self.score_neg))
        
        return loss
    
                 
          
class TransR(KGE):
    """
    Learning Entity and Relation Embeddings for Knowledge Graph Completion.
    """
    
    def __init__(self, args):
        super().__init__(args)
            
        
    def _em_structure(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        P_table = tf.nn.l2_normalize(tf.get_variable('projection_table',
                  [self.n_R, self.dim, self.dim],
                  initializer = self.initializer), 1)
        p_pos = tf.gather(P_table, self.T_pos[:, 1])
        p_neg = tf.gather(P_table, self.T_neg[:, 1])
        
        h_pos = tf.matmul(h_pos, p_pos) 
        t_pos = tf.matmul(t_pos, p_pos)
        h_neg = tf.matmul(h_neg, p_neg)
        t_neg = tf.matmul(t_neg, p_neg)
                    
        s_pos = h_pos + r_pos - t_pos
        s_neg = h_neg + r_neg - t_neg
        
        return s_pos, s_neg
    
    
    def _cal_score(self, s_pos, s_neg):
        score_pos = tf.reduce_sum(s_pos ** 2, [1, 2])
        score_neg = tf.reduce_sum(s_neg ** 2, [1, 2])
        
        return score_pos, score_neg
        
    
    def _cal_loss(self):
        loss = tf.reduce_sum(tf.nn.relu(self.margin + self.score_pos - \
                                        self.score_neg))
        
        return loss
    
    

class TransD(KGE):
    """Knowledge Graph Embedding via Dynamic Mapping Matrix."""
    
    def __init__(self, args):
        super().__init__(args)
            
        
    def _em_structure(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        _projector = lambda s, p_e, p_r: \
            s + tf.reduce_sum(p_e * s, 1, keepdims = True) * p_r
        P_E_table = tf.nn.l2_normalize(tf.get_variable( \
                    'projection_entity_table', [self.n_E, self.dim],
                    initializer = self.initializer), 1)
        p_h_pos = tf.gather(P_E_table, self.T_pos[:, 0])
        p_t_pos = tf.gather(P_E_table, self.T_pos[:, 2])
        p_h_neg = tf.gather(P_E_table, self.T_neg[:, 0])
        p_t_neg = tf.gather(P_E_table, self.T_neg[:, 2])
        
        P_R_table = tf.nn.l2_normalize(tf.get_variable( \
                    'projection_relation_table', [self.n_R, self.dim],
                    initializer = self.initializer), 1)
        p_r_pos = tf.gather(P_R_table, self.T_pos[:, 1])
        p_r_neg = tf.gather(P_R_table, self.T_neg[:, 1])
        
        h_pos = _projector(h_pos, p_h_pos, p_r_pos)
        t_pos = _projector(t_pos, p_t_pos, p_r_pos)
        h_neg = _projector(h_neg, p_h_neg, p_r_neg)
        t_neg = _projector(t_neg, p_t_neg, p_r_neg)
                    
        s_pos = h_pos + r_pos - t_pos
        s_neg = h_neg + r_neg - t_neg
        
        return s_pos, s_neg
    
    
    def _cal_score(self, s_pos, s_neg):
        score_pos = tf.reduce_sum(s_pos ** 2, 1)
        score_neg = tf.reduce_sum(s_neg ** 2, 1)
        
        return score_pos, score_neg
        
    
    def _cal_loss(self):
        loss = tf.reduce_sum(tf.nn.relu(self.margin + self.score_pos - \
                                        self.score_neg))
        
        return loss
    
                        
    
class ConvKB(KGE):
    """
    A Novel Embedding Model for Knowledge Base Completion Based on 
    Convolutional Neural Network.
    """
    
    def __init__(self, args):  
        super().__init__(args)
        
        
    def _em_structure(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        init_f = np.array([[[np.random.normal(0.1, 0.01, self.n_filter)],  
                            [np.random.normal(0.1, 0.01, self.n_filter)], 
                            [np.random.normal(-0.1, 0.01, self.n_filter)]]])
        f = tf.get_variable('filter', [1, 3, 1, self.n_filter],
            initializer = tf.constant_initializer(init_f))
        
        #(B, D, 3, 1) conv (1, 3, 1, F) ==> (B, D, 1, F)
        s_pos = tf.nn.conv2d(tf.concat([h_pos, r_pos, t_pos], 2), 
                f, strides = [1, 1, 1, 1], padding = 'VALID') 
        s_neg = tf.nn.conv2d(tf.concat([h_neg, r_neg, t_neg], 2), 
                f, strides = [1, 1, 1, 1], padding = 'VALID')
        
        return s_pos, s_neg
    
    
    def _cal_score(self, s_pos, s_neg):
        w = tf.get_variable('weight', [self.dim * self.n_filter, 1],
                            initializer = self.initializer)
        
        #((B, D, 1, F) ==> (B, D * F)) * (D * F, 1)
        score_pos = tf.squeeze(tf.matmul(tf.nn.relu(tf.reshape(s_pos, 
                               [-1, self.dim * self.n_filter])), w))
        score_neg = tf.squeeze(tf.matmul(tf.nn.relu(tf.reshape(s_neg,
                               [-1, self.dim * self.n_filter])), w))
        self.l2_loss = tf.reduce_sum(w ** 2)
        
        return score_pos, score_neg
        
    
    def _cal_loss(self):
        loss = tf.reduce_sum(tf.nn.softplus(self.score_pos) + \
                             tf.nn.softplus(- self.score_neg)) + \
               0.001 / 2 * self.l2_loss
        
        return loss
            
