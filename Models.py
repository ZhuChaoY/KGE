from KGE import KGE
import numpy as np
import tensorflow.compat.v1 as tf


class TransE(KGE):
    """Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, args):
        super().__init__(args)
            
    
    def em_structure(self, h_pos, t_pos, h_neg, t_neg, r):
        s_pos = h_pos + r - t_pos
        s_neg = h_neg + r - t_neg
        
        return s_pos, s_neg
    
    
    def cal_score(self, s_pos, s_neg):
        score_pos = tf.reduce_sum(s_pos ** 2, 1)
        score_neg = tf.reduce_sum(s_neg ** 2, 1)
        
        return score_pos, score_neg

            

class TransH(KGE):
    """Knowledge Graph Embedding by Translating on Hyperplanes."""
    
    def __init__(self, args):
        super().__init__(args)
            
        
    def em_structure(self, h_pos, t_pos, h_neg, t_neg, r):
        projector = lambda s, p: \
            s - tf.reduce_sum(p * s, 1, keepdims = True) * p
        
        P_table = tf.nn.l2_normalize(tf.get_variable('projection_table',
                  initializer = tf.random_uniform([self.n_R, self.dim],
                                                  -self.K, self.K)), 1)
        p = tf.gather(P_table, self.T_pos[:, 1])
        
        self.l2_s.append(p)
        
        h_pos = projector(h_pos, p)
        t_pos = projector(t_pos, p)
        h_neg = projector(h_neg, p)
        t_neg = projector(t_neg, p)
                    
        s_pos = h_pos + r - t_pos
        s_neg = h_neg + r - t_neg
        
        return s_pos, s_neg
    
    
    def cal_score(self, s_pos, s_neg):
        score_pos = tf.reduce_sum(s_pos ** 2, 1)
        score_neg = tf.reduce_sum(s_neg ** 2, 1)
        
        return score_pos, score_neg
    
    

class TransD(KGE):
    """Knowledge Graph Embedding via Dynamic Mapping Matrix."""
    
    def __init__(self, args):
        super().__init__(args)
            
        
    def em_structure(self, h_pos, t_pos, h_neg, t_neg, r):
        projector = lambda s, p_e, p_r: \
            s + tf.reduce_sum(p_e * s, 1, keepdims = True) * p_r
        P_E_table = tf.nn.l2_normalize(tf.get_variable('projection_entity_' \
                    'table', initializer = tf.random_uniform([self.n_E,
                    self.dim], -self.K, self.K)), 1)
        p_h_pos = tf.gather(P_E_table, self.T_pos[:, 0])
        p_t_pos = tf.gather(P_E_table, self.T_pos[:, -1])
        p_h_neg = tf.gather(P_E_table, self.T_neg[:, 0])
        p_t_neg = tf.gather(P_E_table, self.T_neg[:, -1])
        
        P_R_table = tf.nn.l2_normalize(tf.get_variable('projection_relation_' \
                    'table', initializer = tf.random_uniform([self.n_R,
                    self.dim], -self.K, self.K)), 1)
        p_r = tf.gather(P_R_table, self.T_pos[:, 1])
        
        self.l2_s.extend([p_h_pos, p_t_pos, p_h_neg, p_t_neg, p_r]) 
        
        h_pos = projector(h_pos, p_h_pos, p_r)
        t_pos = projector(t_pos, p_t_pos, p_r)
        h_neg = projector(h_neg, p_h_neg, p_r)
        t_neg = projector(t_neg, p_t_neg, p_r)
                    
        s_pos = h_pos + r - t_pos
        s_neg = h_neg + r - t_neg
        
        return s_pos, s_neg
    
    
    def cal_score(self, s_pos, s_neg):
        score_pos = tf.reduce_sum(s_pos ** 2, 1)
        score_neg = tf.reduce_sum(s_neg ** 2, 1)
        
        return score_pos, score_neg
    
    
    
class ConvKB(KGE):
    """
    A Novel Embedding Model for Knowledge Base Completion Based on 
    Convolutional Neural Network.
    """
    
    def __init__(self, args):  
        super().__init__(args)
        
        
    def em_structure(self, h_pos, t_pos, h_neg, t_neg, r):
        init_f = np.array([[[np.random.normal(0.1, 0.01, self.n_filter)],  
                            [np.random.normal(0.1, 0.01, self.n_filter)], 
                            [np.random.normal(-0.1, 0.01, self.n_filter)]]])
        f = tf.get_variable('filter', [1, 3, 1, self.n_filter],
            initializer = tf.constant_initializer(init_f))
        
        #(B, D, 3, 1) conv (1, 3, 1, F) ==> (B, D, 1, F)
        s_pos = tf.nn.conv2d(tf.concat([h_pos, r, t_pos], 2), 
                f, strides = [1, 1, 1, 1], padding = 'VALID') 
        s_neg = tf.nn.conv2d(tf.concat([h_neg, r, t_neg], 2), 
                f, strides = [1, 1, 1, 1], padding = 'VALID')
        
        return s_pos, s_neg
    
    
    def cal_score(self, s_pos, s_neg):
        dn = self.dim * self.n_filter
        K = np.sqrt(6.0 / (dn + 1))
        w = tf.get_variable('weight', initializer = tf.random_uniform([dn, 1],
                                                                      -K, K))
        
        #((B, D, 1, F) ==> (B, D * F)) * (D * F, 1)
        score_pos = tf.squeeze(tf.matmul(tf.nn.relu(tf.reshape(s_pos, 
                                                               [-1, dn])), w))
        score_neg = tf.squeeze(tf.matmul(tf.nn.relu(tf.reshape(s_neg, 
                                                               [-1, dn])), w))
        self.l2_s.append(w) 
        
        return score_pos, score_neg
    