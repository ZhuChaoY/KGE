from KGE import KGE
import numpy as np
import tensorflow as tf


class TransE(KGE):
    """Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, args):
        super().__init__(args)
            
    
    def kge_variables(self):
        pass
    
        
    def em_structure(self, h, r, t, key = 'pos'):
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
        self.p_pos = tf.gather(P_table, self.T_pos[:, 1])
        self.p_neg = tf.reshape(tf.tile(tf.expand_dims(self.p_pos, 1), 
                     [1, self.n_neg, 1]), [-1, self.dim])
        
        self.l2_kge.append(self.p_pos)
        
        
    def em_structure(self, h, r, t, key = 'pos'):       
        self.projector = lambda s, p: \
            s - tf.reduce_sum(p * s, -1, keepdims = True) * p    
            
        if key == 'pos':
            p = self.p_pos
        elif key == 'neg':
            p = self.p_neg
            
        h = self.projector(h, p)
        t = self.projector(t, p)
        return h + r - t
    
    
    def cal_score(self, s):
        return tf.reduce_sum(tf.abs(s), -1)
    
    
    def cal_lp_score(self, h, r, t):        
        p = self.p_pos
        p_E_table = self.projector(self.E_table, tf.expand_dims(p, 1))
        s_rpc_h = p_E_table + tf.expand_dims(r - self.projector(t, p), 1)
        s_rpc_t = tf.expand_dims(self.projector(h, p) + r, 1) - p_E_table
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
    
        
    def em_structure(self, h, r, t, key = 'pos'):
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
        
        
        
# class TransD(KGE):
#     """Knowledge Graph Embedding via Dynamic Mapping Matrix."""
    
#     def __init__(self, args):
#         super().__init__(args)
            
    
#     def kge_variables(self):
#         self.PE_table = tf.get_variable('projection_entity_table',
#                         initializer = tf.random_uniform([self.n_E, self.dim],
#                                                         -self.K, self.K))
#         self.PE_table = tf.nn.l2_normalize(self.PE_table, 1)
#         self.ph_pos = tf.gather(self.PE_table, self.T_pos[:, 0])
#         self.pt_pos = tf.gather(self.PE_table, self.T_pos[:, -1])
#         self.ph_neg = tf.gather(self.PE_table, self.T_neg[:, 0])
#         self.pt_neg = tf.gather(self.PE_table, self.T_neg[:, -1])
        
#         PR_table = tf.get_variable('projection_relation_table', initializer = \
#                    tf.random_uniform([self.n_R, self.dim], -self.K, self.K))
#         PR_table = tf.nn.l2_normalize(PR_table, 1)
#         self.pr_pos = tf.gather(PR_table, self.T_pos[:, 1])
#         self.pr_neg = tf.reshape(tf.tile(tf.expand_dims(self.pr_pos, 1), 
#                       [1, self.n_neg, 1]), [-1, self.dim])
        
#         self.l2_kge.extend([self.ph_pos, self.pt_pos, self.pr_pos,
#                             self.ph_neg, self.pt_neg]) 
            
        
#     def em_structure(self, h, r, t, key = 'pos'):
#         self.projector = lambda s, pe, pr: \
#             s + tf.reduce_sum(pe * s, -1, keepdims = True) * pr 
        
#         if key == 'pos':
#             ph, pr, pt = self.ph_pos, self.pr_pos, self.pt_pos
#         elif key == 'neg':
#             ph, pr, pt = self.ph_neg, self.pr_neg, self.pt_neg
            
#         h = self.projector(h, ph, pr)
#         t = self.projector(t, pt, pr)
#         return h + r - t
    
    
#     def cal_score(self, s):
#         return tf.reduce_sum(tf.abs(s), -1)
    
    
#     def cal_lp_score(self, h, r, t):    
#         ph, pr, pt = self.ph_pos, self.pr_pos, self.pt_pos
#         p_E_table = self.projector(self.E_table, self.PE_table,
#                                    tf.expand_dims(pr, 1))
#         s_rpc_h = p_E_table + tf.expand_dims(r - self.projector(t, pt, pr), 1)
#         s_rpc_t = tf.expand_dims(self.projector(h, ph, pr) + r, 1) - p_E_table
#         return self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        