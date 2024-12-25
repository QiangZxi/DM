import torch
from torch import nn

from .base import BasePromptLearner, _TEXT_DIM, _VISION_DIM, _get_clones
from .multimodal_mixture import MultiheadAttention, LinearMultimodalMixture, CrossAttentionMultimodalMixture


class ProgressivePromptLearner(BasePromptLearner):
    """
    递进式的prompt learner策略
    """
    def __init__(self, 
                 cfg, 
                 classnames, 
                 clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.progressive_proj_dim = cfg.TRAINER.PMAPLE.PROGRESSIVE_PROJ_DIM
        self._init_progressive_projs()
    
    def _forward_deep_prompts(self):
        deep_prompts_list_text = [self.deep_prompts_text]
        for proj in self.progressive_projs:
            deep_prompts_text = deep_prompts_list_text[-1]
            deep_prompts_list_text.append(deep_prompts_text + proj(deep_prompts_text))
        
        deep_prompts_list_visual = []
        for idx, layer in enumerate(self.deep_projs):
            deep_prompts_list_visual.append(layer(deep_prompts_list_text[idx]))
        return deep_prompts_list_text, deep_prompts_list_visual
    
    def _init_deep_prompts_text(self):
        self.deep_prompts_text = nn.Parameter(torch.empty(self.prompt_len, _TEXT_DIM))
        nn.init.normal_(self.deep_prompts_text, std=0.02) 
    
    def _init_progressive_projs(self):
        down = nn.Linear(_TEXT_DIM, self.progressive_proj_dim)
        up = nn.Linear(self.progressive_proj_dim, _TEXT_DIM)
        proj = nn.Sequential(down, nn.ReLU(), up)
        self.progressive_projs = _get_clones(proj, self.deep_prompts_depth - 1)


class BiPromptLearner(BasePromptLearner):
    """
    双向的prompt learner策略
    """
    def __init__(self, 
                 cfg, 
                 classnames, 
                 clip_model):
        self.mode = cfg.TRAINER.BMAPLE.MODE
        self.cross_mode = cfg.TRAINER.BMAPLE.CROSS_MODE
        self.attn_dim = cfg.TRAINER.BMAPLE.ATTN_DIM
        self.num_heads = cfg.TRAINER.BMAPLE.NUM_HEADS
        self.residual = cfg.TRAINER.BMAPLE.RESIDUAL
        self.share_weights = cfg.TRAINER.BMAPLE.SHARE_WEIGHTS
        super().__init__(cfg, classnames, clip_model)
        
    def _forward_deep_prompts(self):
        deep_inputs_list_text, deep_inputs_list_visual = self.deep_prompts_list_text, self.deep_prompts_list_vision
        deep_prompts_list_text, deep_prompts_list_visual = [], []
        
        for idx, layer in enumerate(self.deep_projs):
            deep_prompts_text, deep_prompts_visual = layer(deep_inputs_list_text[idx], deep_inputs_list_visual[idx])
            deep_prompts_list_text.append(deep_prompts_text)
            deep_prompts_list_visual.append(deep_prompts_visual)

        return deep_prompts_list_text, deep_prompts_list_visual
    
    def _init_deep_prompts_vision(self):
        self.deep_prompts_list_vision = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_len, _VISION_DIM)) 
                                                          for _ in range(self.deep_prompts_depth - 1)])
        for p in self.deep_prompts_list_vision:
            nn.init.normal_(p, std=0.02)     
    
    def _init_deep_projs(self):
        assert self.mode in ['linear', 'attn'], f'mode {self.mode} is not supported!'
        if self.mode == 'linear':
            proj = LinearMultimodalMixture(self.residual, self.share_weights)
        else:
            proj = CrossAttentionMultimodalMixture(self.attn_dim, self.num_heads, self.cross_mode,
                                                   self.residual, self.share_weights)
        self.deep_projs = _get_clones(proj, self.deep_prompts_depth - 1)


class CrossLayerPromptLearner(BasePromptLearner):
    """
    跨层次的prompt learner策略
    """
    def __init__(self, 
                 cfg, 
                 classnames, 
                 clip_model):
        self.attn_dim = cfg.TRAINER.CMAPLE.ATTN_DIM
        self.num_heads = cfg.TRAINER.CMAPLE.NUM_HEADS
        self.use_mask = cfg.TRAINER.CMAPLE.USE_MASK
        self.use_positional_embedding = cfg.TRAINER.CMAPLE.USE_POSITIONAL_EMBEDDING
        self.use_layer_embedding = cfg.TRAINER.CMAPLE.USE_LAYER_EMBEDDING
        super().__init__(cfg, classnames, clip_model)
        
        if self.use_positional_embedding:
            self.pos_embed = nn.Parameter(torch.empty(self.prompt_len, 1))
            nn.init.normal_(self.pos_embed, std=0.01)
        if self.use_layer_embedding:
            self.layer_embed = nn.Parameter(torch.empty(self.deep_prompts_depth - 1, 1))
            nn.init.normal_(self.layer_embed, std=0.01)

    def _forward_deep_prompts(self):
        # [(N, D)] -> (L, N, D) -> (L * N, D)
        deep_prompts_list_text = [p for p in self.deep_prompts_list_text]

        deep_prompts_text = torch.stack(deep_prompts_list_text, dim=0)
        
        # L: num_layers, N: prompt_len
        L,  N = deep_prompts_text.size()[:-1]
        deep_prompts_text = deep_prompts_text.reshape(L * N, -1)
        
        if self.use_positional_embedding:
            deep_prompts_text = deep_prompts_text + self.pos_embed.unsqueeze(0) \
                                                        .repeat(L, 1, _TEXT_DIM).reshape(-1, _TEXT_DIM)
        if self.use_layer_embedding:
            deep_prompts_text = deep_prompts_text + self.layer_embed.unsqueeze(1) \
                                                        .repeat(1, N, _TEXT_DIM).reshape(-1, _TEXT_DIM)

        mask = self._gen_mask(L, N).to(deep_prompts_text.device) if self.use_mask else None
        deep_prompts_vision = self.deep_proj(deep_prompts_text, deep_prompts_text, deep_prompts_text, mask=mask)

        deep_prompts_list_vision = deep_prompts_vision.reshape(L, N, -1)
        return deep_prompts_list_text, deep_prompts_list_vision
        
    def _gen_mask(self, L, N):
        mask = torch.zeros((L * N, L * N)).bool()
        for row in range(L):
            for col in range(L):
                if row < col:
                    continue
                mask[row * N:(row + 1) * N, col * N:(col + 1) * N] = True
        return mask

    def _init_deep_projs(self):
        self.deep_proj = MultiheadAttention(_TEXT_DIM, _TEXT_DIM, _TEXT_DIM, _VISION_DIM, 
                                            self.attn_dim, self.num_heads)


class ProgressiveBiPromptLearner(BasePromptLearner):
    """
    递进式双向的prompt learner策略
    """
    def __init__(self, 
                 cfg, 
                 classnames, 
                 clip_model):
        self.mode = cfg.TRAINER.PBMAPLE.MODE
        self.cross_mode = cfg.TRAINER.PBMAPLE.CROSS_MODE
        self.attn_dim = cfg.TRAINER.PBMAPLE.ATTN_DIM
        self.num_heads = cfg.TRAINER.PBMAPLE.NUM_HEADS
        self.residual = cfg.TRAINER.PBMAPLE.RESIDUAL
        self.share_weights = cfg.TRAINER.PBMAPLE.SHARE_WEIGHTS
        super().__init__(cfg, classnames, clip_model)
        
    def _forward_deep_prompts(self):
        deep_prompts_list_text = [self.deep_prompts_text]
        deep_prompts_list_vision = [self.deep_prompts_vision]

        for idx, proj in enumerate(self.deep_projs):
            inputs_text, inputs_vision = deep_prompts_list_text[-1], deep_prompts_list_vision[-1]
            prompts_text, prompts_vision = proj(inputs_text, inputs_vision)    
            deep_prompts_list_text.append(prompts_text)
            deep_prompts_list_vision.append(prompts_vision)
        
        return deep_prompts_list_text, deep_prompts_list_vision

    def _init_deep_prompts_text(self):
        self.deep_prompts_text = nn.Parameter(torch.empty(self.prompt_len, _TEXT_DIM))
        nn.init.normal_(self.deep_prompts_text, std=0.02)
    
    def _init_deep_prompts_vision(self):
        self.deep_prompts_vision = nn.Parameter(torch.empty(self.prompt_len, _VISION_DIM))
        nn.init.normal_(self.deep_prompts_vision, std=0.02)

    def _init_deep_projs(self):
        assert self.mode in ['linear', 'attn'], f'mode {self.mode} is not supported!'
        if self.mode == 'linear':
            proj = LinearMultimodalMixture(self.residual, self.share_weights)
        else:
            proj = CrossAttentionMultimodalMixture(self.attn_dim, self.num_heads, self.cross_mode,
                                                   self.residual, self.share_weights)
        self.deep_projs = _get_clones(proj, self.deep_prompts_depth - 1)
