from KGE import KGE
import numpy as np
import tensorflow as tf


class TransE(KGE):
    """Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, args):
        super().__init__(args)
            
    
    def kge_variables(self):
        pass
    
        
    def em_structure(self, h, r, t, key):
        return h + r - t
    
    
    def cal_score(self, s):
        return tf.reduce_sum(tf.abs(s), -1)
            
    
    def cal_lp_score(self, h, r, t):        
        s_rpc_h = self.E_table + tf.expand_dims(r - t, 1)  
        s_rpc_t = tf.expand_dims(h + r, 1) - self.E_table
        return self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        
            

class TransH(KGE):
    """Knowledge Graph Embedding by Translating on Hyperplanes."""
    
    def __init__(self, args):
        super().__init__(args)
        
            
    def kge_variables(self):
        P_table = tf.get_variable('projection_table', initializer = \
                  tf.random_uniform([self.n_R, self.dim], -self.K, self.K))
        P_table = tf.nn.l2_normalize(P_table, 1)
        self.p = tf.gather(P_table, self.T_pos[:, 1])

        self.l2_kge.append(self.p)
        
        
    def em_structure(self, h, r, t, key):       
        self.projector = lambda s, p: \
            s - tf.reduce_sum(p * s, -1, keepdims = True) * p    
            
        h = self.projector(h, self.p)
        t = self.projector(t, self.p)
        return h + r - t
    
    
    def cal_score(self, s):
        return tf.reduce_sum(tf.abs(s), -1)
    
    
    def cal_lp_score(self, h, r, t):        
        p_E_table = self.projector(self.E_table, tf.expand_dims(self.p, 1))
        s_rpc_h = p_E_table + tf.expand_dims(r - self.projector(t, self.p), 1)
        s_rpc_t = tf.expand_dims(self.projector(h, self.p) + r, 1) - p_E_table
        return self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
    

    
class ConvKB(KGE):
    """
    A Novel Embedding Model for Knowledge Base Completion Based on 
    Convolutional Neural Network.
    """
    
    def __init__(self, args):  
        super().__init__(args)
        
        
    def kge_variables(self):
        init_f = np.array([[[np.random.normal(0.1, 0.01, self.n_filter)],  
                            [np.random.normal(0.1, 0.01, self.n_filter)], 
                            [np.random.normal(-0.1, 0.01, self.n_filter)]]])
        self.F = tf.get_variable('filter', [1, 3, 1, self.n_filter],
                                 initializer = tf.constant_initializer(init_f))
        
        K = np.sqrt(6.0 / (self.dim * self.n_filter + 1))
        self.W = tf.get_variable('weight', initializer = \
                 tf.random_uniform([self.dim * self.n_filter, 1], -K, K))
        
        self.l2_kge.append(self.W)
    
        
    def em_structure(self, h, r, t, key):
        #(B, D, 3, 1) conv (1, 3, 1, F) ==> (B, D, 1, F)
        return tf.nn.conv2d(tf.concat([h, r, t], 2), self.F,
                            strides = [1, 1, 1, 1], padding = 'VALID') 
    
    
    def cal_score(self, s):
        #((B, D, 1, F) ==> (B, D * F)) * (D * F, 1) ==> (B, 1)
        return tf.matmul(tf.nn.relu(tf.reshape(s, [-1, self.dim * \
                                                   self.n_filter])), self.W)
      
        
    def cal_lp_score(self, h, r, t):                 
        s_rpc_h = tf.nn.conv2d(tf.concat([self.E_table, tf.tile( \
                  tf.concat([r, t], 2), [self.n_E, 1, 1, 1])], 2), self.F,
                  strides = [1, 1, 1, 1], padding = 'VALID')
        s_rpc_t = tf.nn.conv2d(tf.concat([tf.tile(tf.concat([h, r], 2),
                  [self.n_E, 1, 1, 1]), self.E_table], 2), self.F,
                  strides = [1, 1, 1, 1], padding = 'VALID')
        return tf.reshape(self.cal_score(s_rpc_h), [1, -1]), \
               tf.reshape(self.cal_score(s_rpc_t), [1, -1])
               
           

class RotatE(KGE):
    """
    ROTATE: KNOWLEDGE GRAPH EMBEDDING BY RELATIONAL ROTATION IN COMPLEX SPACE.
    """
    
    def __init__(self, args):
        super().__init__(args)
        
            
    def kge_variables(self):
        self.E_I_table = tf.get_variable('entity_imaginary_table',
                         initializer = tf.random_uniform([self.n_E, self.dim],
                                                          -self.K, self.K))
        self.E_I_table = tf.nn.l2_normalize(self.E_I_table, 1)
        self.h_i_pos = tf.gather(self.E_I_table, self.T_pos[:, 0])
        self.t_i_pos = tf.gather(self.E_I_table, self.T_pos[:, -1])
        self.h_i_neg = tf.gather(self.E_I_table, self.T_neg[:, 0])
        self.t_i_neg = tf.gather(self.E_I_table, self.T_neg[:, -1])
        
        self.l2_kge.extend([self.h_i_pos, self.t_i_pos, 
                            self.h_i_neg, self.t_i_neg])
        
        
    def em_structure(self, h, r, t, key):               
        r_r, r_i = tf.cos(r), tf.sin(r)
        if key == 'pos':
            h_i, t_i = self.h_i_pos, self.t_i_pos
        else:
            h_i, t_i = self.h_i_neg, self.t_i_neg
        
        re = h * r_r - h_i * r_i - t
        im = h * r_i + h_i * r_r - t_i
        return tf.concat([re, im], -1)
    
    
    def cal_score(self, s):
        return tf.reduce_sum(tf.abs(s), -1)
    
    
    def cal_lp_score(self, h, r, t):     
        r_r, r_i = tf.cos(r), tf.sin(r)
        
        s_rpc_h_r = self.E_table * tf.expand_dims(r_r, 1) - self.E_I_table * \
                    tf.expand_dims(r_i, 1) - tf.expand_dims(t, 1)
        s_rpc_h_i = self.E_table * tf.expand_dims(r_i, 1) + self.E_I_table * \
                    tf.expand_dims(r_r, 1) - tf.expand_dims(self.t_i_pos, 1)
        s_rpc_h = tf.concat([s_rpc_h_r, s_rpc_h_i], -1)
        
        s_rpc_t_r = tf.expand_dims(h * r_r - self.h_i_pos * r_i, 1) - \
                    self.E_table
        s_rpc_t_i = tf.expand_dims(h * r_i + self.h_i_pos * r_r, 1) - \
                    self.E_I_table
        s_rpc_t = tf.concat([s_rpc_t_r, s_rpc_t_i], -1)
        
        return self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)