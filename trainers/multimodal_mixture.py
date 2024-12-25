import torch
import torch.nn.functional as F
from torch import nn


_TEXT_DIM = 512
_VISION_DIM = 768


class MultiheadAttention(nn.Module):
    def __init__(self, 
                 q_dim, 
                 k_dim, 
                 v_dim, 
                 out_dim,
                 attn_dim, 
                 num_heads):
        super().__init__()
        assert attn_dim % num_heads == 0

        self.num_heads = num_heads
        self.scale = (attn_dim / num_heads) ** -0.5
        
        self.wq = nn.Linear(q_dim, attn_dim)
        self.wk = nn.Linear(k_dim, attn_dim)
        self.wv = nn.Linear(v_dim, attn_dim)
        self.wo = nn.Linear(attn_dim, out_dim)
        
    def forward(self, q, k, v, mask=None):
        """
        Params:
        - q Tensor(*, N, Dq)
        - k Tensor(*, N, Dk)
        - v Tensor(*, N, Dv)
        """
        assert len(q.size()) in [2, 3], f'size of query {q.size()} is not supported!'
        # (*, N, D*) -> (*, N, D)
        q = self.wq(q) 
        k = self.wk(k)
        v = self.wv(v)

        has_batch = False if len(q.size()) == 2 else True
        num_heads = self.num_heads
        
        Nq = q.size(-2)
        Nk = k.size(-2)
        Nv = v.size(-2)

        if not has_batch:
            # (N, D) -> (N , H, D // H) -> (H, N, D // H)
            q = q.view(Nq, num_heads, -1).permute(1, 0, 2)
            k = k.view(Nk, num_heads, -1).permute(1, 0, 2)
            v = v.view(Nv, num_heads, -1).permute(1, 0, 2)
        else:
            # (B, N, D) -> (B, N, H, D // H) -> (B, H, N, D // H) -> (B * H, N, D // H)
            B = q.size(0)
            q = q.view(B, Nq, num_heads, -1).permute(0, 2, 1, 3).reshape(B * num_heads, Nq, -1)
            k = k.view(B, Nk, num_heads, -1).permute(0, 2, 1, 3).reshape(B * num_heads, Nk, -1)
            v = v.view(B, Nv, num_heads, -1).permute(0, 2, 1, 3).reshape(B * num_heads, Nv, -1)

        # (*, N, D // H) @ (*, D // H, N) -> (*, N, N)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(0).repeat(num_heads, 1, 1)
            attn = attn.masked_fill(mask, -1e6)

        # (*, N, N) @ (*, N, D // H) -> (*, N, D // H)
        attn = attn.softmax(dim=-1)
        o = torch.bmm(attn, v)

        if not has_batch:
            # (H, N, D // H) -> (N, H, D // H) -> (N, D)
            o = o.permute(1, 0, 2).reshape(Nq, -1)
        else:
            # (B * H, N, D // H) -> (B, H, N, D // H) -> (B, N, H, D // H) -> (B, N, D)
            o = o.reshape(B, num_heads, Nq, -1).permute(0, 2, 1, 3).reshape(B, num_heads, -1)

        o = self.wo(o)
        return o

        
class MultiHeadDualAttention(nn.Module):
    def __init__(self, 
                 k1_dim, 
                 v1_dim, 
                 k2_dim, 
                 v2_dim, 
                 o1_dim,
                 o2_dim,
                 attn_dim,
                 num_heads):
        super().__init__()
        assert attn_dim % num_heads == 0

        self.num_heads = num_heads
        self.scale = (attn_dim / num_heads) ** -0.5

        self.wk1 = nn.Linear(k1_dim, attn_dim)
        self.wv1 = nn.Linear(v1_dim, attn_dim)
        self.wk2 = nn.Linear(k2_dim, attn_dim)
        self.wv2 = nn.Linear(v2_dim, attn_dim)
        self.wo1 = nn.Linear(attn_dim, o1_dim)
        self.wo2 = nn.Linear(attn_dim, o2_dim)
        
    def forward(self, k1, v1, k2, v2):
        k1 = self.wk1(k1)
        v1 = self.wv1(v1)
        k2 = self.wk2(k2)
        v2 = self.wv2(v2)
        
        N = k1.size(0)
        num_heads = self.num_heads

        # (N, D') -> (N, H, D' // H) -> (H, N, D' // H)
        k1 = k1.view(N, num_heads, -1).permute(1, 0, 2)
        v1 = v1.view(N, num_heads, -1).permute(1, 0, 2)
        k2 = k2.view(N, num_heads, -1).permute(1, 0, 2)
        v2 = v2.view(N, num_heads, -1).permute(1, 0, 2)

        # (H, N, D' // H) @ (H, D' // H, N) -> (H, N, N)
        attn = torch.bmm(k1, k2.transpose(1, 2)) * self.scale
        # (H, N, N) @ (H, N, D' // H) -> (H, N, D' // H)
        o1 = torch.bmm(attn.transpose(1, 2).softmax(dim=-1), v1)
        o2 = torch.bmm(attn.softmax(dim=-1), v2)

        # (H, N, D' // H) -> (N, H, D' // H) -> (N, D')
        o1 = o1.permute(1, 0, 2).reshape(N, -1)
        o2 = o2.permute(1, 0, 2).reshape(N, -1)
        o1 = self.wo1(o1)
        o2 = self.wo2(o2)

        return o1, o2


class LinearMultimodalMixture(nn.Module):
    """
    线性的多模态混合策略
    """
    def __init__(self, 
                 residual=True,
                 share_weights=False):
        super().__init__()
        self.residual = residual
        self.share_weights = share_weights
        
        if share_weights:
            self.proj_weight = nn.Parameter(torch.empty(_TEXT_DIM, _VISION_DIM))
            self.proj_bias_t = nn.Parameter(torch.zeros(_TEXT_DIM, ))
            self.proj_bias_v = nn.Parameter(torch.zeros(_VISION_DIM, ))
            nn.init.normal_(self.proj_weight, std=0.02)
        else:
            self.proj_t2v = nn.Linear(_TEXT_DIM, _VISION_DIM)
            self.proj_v2t = nn.Linear(_VISION_DIM, _TEXT_DIM)
    
    def forward(self, inputs_text, inputs_vision):
        """
        Params:
        - inputs_text    Tensor(N, DT): 输入的文本向量
        - inputs_vision  Tensor(N, DV): 输入的视觉向量
        Return:
        - prompts_text   Tensor(N, DT): 文本prompt
        - prompts_vision Tensor(N, DV): 视觉prompt
        """
        if self.share_weights:
            prompts_text = F.linear(inputs_vision, self.proj_weight, self.proj_bias_t)
            prompts_vision = F.linear(inputs_text, self.proj_weight.t(), self.proj_bias_v)
        else:
            prompts_text = self.proj_v2t(inputs_vision)
            prompts_vision = self.proj_t2v(inputs_text)

        if self.residual:
            prompts_text = prompts_text + inputs_text
            prompts_vision = prompts_vision + inputs_vision
            
        return prompts_text, prompts_vision


class CrossAttentionMultimodalMixture(nn.Module):
    """
    交叉注意力的多模态混合策略
    """
    def __init__(self, 
                 attn_dim=16,
                 num_heads=1,
                 mode='cross_q',
                 residual=True,
                 share_weights=False):
        super().__init__()
        assert mode in ['cross_q', 'cross_kv']
        self.mode = mode
        self.residual = residual
        self.share_weights = share_weights

        if share_weights:
            self.attn = MultiHeadDualAttention(_TEXT_DIM, _TEXT_DIM, _VISION_DIM, _VISION_DIM, 
                                               _TEXT_DIM, _VISION_DIM, attn_dim, num_heads)
        else:
            if mode == 'cross_q':
                self.attn_t2v = MultiheadAttention(_TEXT_DIM, _VISION_DIM, _VISION_DIM, 
                                                   _VISION_DIM, attn_dim, num_heads)    
                self.attn_v2t = MultiheadAttention(_VISION_DIM, _TEXT_DIM, _TEXT_DIM, 
                                                   _TEXT_DIM, attn_dim, num_heads)   
            else:
                self.attn_t2v = MultiheadAttention(_VISION_DIM, _TEXT_DIM, _TEXT_DIM, 
                                                   _VISION_DIM, attn_dim, num_heads)    
                self.attn_v2t = MultiheadAttention(_TEXT_DIM, _VISION_DIM, _VISION_DIM, 
                                                   _TEXT_DIM, attn_dim, num_heads)   
    
    def forward(self, inputs_text, inputs_vision):
        """
        Params:
        - inputs_text    Tensor(N, Lt, Dt): 输入的文本向量
        - inputs_vision  Tensor(B, Lv, Dv): 输入的视觉向量
        Return:
        - prompts_text   Tensor(N, Lt, Dt): 文本prompt
        - prompts_vision Tensor(B, Lv, Dv): 视觉prompt
        """
        if self.share_weights:
            raise NotImplementedError()
        else:
            has_batch = True if len(inputs_text.size()) == 3 else False
            if self.mode == 'cross_q' and has_batch:
                raise NotImplementedError('cross_q cannot be with batched inputs!')
            
            if has_batch:
                N, Lt = inputs_text.size()[:-1]
                B, Lv = inputs_vision.size()[:-1]
                # (N, Lt, Dt) -> (N, B, Lt, Dt) -> (N * B, Lt, Dt)
                # (B, Lv, Dv) -> (N, B, Lv, Dv) -> (N * B, Lv, Dv)
                inputs_text = inputs_text.unsqueeze(1).repeat(1, B, 1, 1).reshape(N * B, Lt, -1)
                inputs_vision = inputs_vision.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N * B, Lv, -1)
                
            if self.mode == 'cross_q':
                prompts_text = self.attn_v2t(inputs_vision, inputs_text, inputs_text)
                prompts_vision = self.attn_t2v(inputs_text, inputs_vision, inputs_vision)
            else:
                prompts_text = self.attn_v2t(inputs_text, inputs_vision, inputs_vision)
                prompts_vision = self.attn_t2v(inputs_vision, inputs_text, inputs_text)
        
            if has_batch:
                # (N * B, Lt, Dt) -> (N, B, Lt, Dt) -> (N, Lt, Dt)
                # (N * B, Lv, Dv) -> (N, B, Lv, Dv) -> (B, Lt, Dt)
                prompts_text = prompts_text.reshape(N, B, Lt, -1).mean(1)
                prompts_vision = prompts_vision.reshape(N, B, Lv, -1).mean(0)
        
        if self.residual:
            prompts_text = prompts_text + inputs_text
            prompts_vision = prompts_vision + inputs_vision
        
        return prompts_text, prompts_vision
