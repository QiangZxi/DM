import copy
import datetime
import time
import torch
from torch import nn

from dassl.engine.trainer import TrainerX as TrainerX_, MetricMeter, AverageMeter, SummaryWriter

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.logger import print


_TEXT_DIM = 512
_VISION_DIM = 768
_tokenizer = _Tokenizer()

def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class BasePromptLearner(nn.Module):
    """
    MaPLe的prompt learner策略
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # 初始化配置
        self._init_cfg(cfg, classnames, clip_model)
        # 初始化输入层prompts
        self._init_prompts_text(clip_model)
        self._init_prompts_vision()
        # 初始化深层prompts
        self._init_deep_prompts_text()
        self._init_deep_prompts_vision()
        # 初始化交互方式
        self._init_proj()
        self._init_deep_projs()
        # 初始化离散形式的prompts, 用于提取文本prompts的前后缀并提供TextEncoder输出层token的索引
        self._init_tokenized_prompts(clip_model)
        if cfg.TRAINER.NAME == 'PromptSRC_pb':
            self._init_srcmodel(clip_model)
    

    def construct_prompts(self, prompts, prefix, suffix, label=None):
        """
        拼接前后缀以构造输入层文本prompts
        Params:
        - prompts Tensor(B, prompt_len, D)    : embedding形式的prompts
        - prefix  Tensor(B, 1, D)             : prompts前缀, 即<SCS>的embedding形式
        - suffix  Tensor(B, N - prompt_len, D): prompts后缀, 即<CLS> + <EOS> + <BLK>的embedding形式
        Return:
        - prompts Tensor(B, N, D)             : 拼接后的完整prompts
        """
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, prompts, suffix], dim=1)
        return prompts

    def forward(self):
        """
        生成输入层和深层的文本/视觉prompts
        """
        prompts_text, prompts_visual = self._forward_prompts()
        deep_prompts_list_text, deep_prompts_list_visual = self._forward_deep_prompts()
        return prompts_text, prompts_visual, deep_prompts_list_text, deep_prompts_list_visual
    
    def _forward_prompts(self):
        """
        生成输入层的prompts
        """
        prompts_text = self.prompts_text

        if prompts_text.dim() == 2:
            # (prompt_len, D) -> (B, prompt_len, D)
            prompts_text = prompts_text.unsqueeze(0).expand(self.num_classes, -1, -1)

        # 拼接<SCS>, <CLS>, <EOS>和<BLK>, 为每个类别组成完整的prompts
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts_text = self.construct_prompts(prompts_text, prefix, suffix)
        prompts_visual = self.proj(self.prompts_text)
        return prompts_text, prompts_visual
    
    def _forward_deep_prompts(self):
        """
        生成深层的prompts
        """
        deep_prompts_list_text = self.deep_prompts_list_text
        deep_prompts_list_visual = []
        for idx, layer in enumerate(self.deep_projs):
            deep_prompts_list_visual.append(layer(deep_prompts_list_text[idx]))
        return deep_prompts_list_text, deep_prompts_list_visual
    
    def _init_cfg(self, cfg, classnames, clip_model):
        """
        一些初始化配置
        """
        self.cfg = cfg
        self.classnames = classnames
        
        self.num_classes = len(classnames)

        self.prompt_init = cfg.TRAINER.MAPLE.CTX_INIT
        self.prompt_len = cfg.TRAINER.MAPLE.N_CTX
        self.prompt_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype

        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "prompt深度至少为1!"
        self.deep_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH 

        clip_imsize = clip_model.visual.input_resolution
        imsize = cfg.INPUT.SIZE[0]
        assert imsize == clip_imsize, f"图像尺寸({imsize})需要与CLIP的输入尺寸({clip_imsize})相等!"
    
    def _init_prompts_text(self, clip_model):
        """
        初始化输入层文本prompts
        """
        prompt_init = self.prompt_init
        prompt_len = self.prompt_len
        prompt_dim = self.prompt_dim
        dtype = self.dtype
        
        if prompt_init and (prompt_len) <= 4:
            # 如果设置了初始化prompts的文本, 例如"a photo of a", 则将该文本的embedding作为输入层的初始prompts
            prompt = clip.tokenize(prompt_init)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            prompt_vectors = embedding[0, 1: 1 + prompt_len, :]
            prompt_prefix = prompt_init
        else:
            # 如果未设置初始化prompts的文本, 则随机初始化一个prompts
            prompt_vectors = torch.empty(prompt_len, prompt_dim, dtype=dtype)
            nn.init.normal_(prompt_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * prompt_len)
        
        self.prompts_text = nn.Parameter(prompt_vectors)
        self.prompt_prefix = prompt_prefix
        
    def _init_prompts_vision(self):
        """
        初始化输入层视觉prompts
        """
        pass

    def _init_deep_prompts_text(self):
        """
        初始化深层文本prompts
        """
        self.deep_prompts_list_text = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_len, _TEXT_DIM)) 
                                                        for _ in range(self.deep_prompts_depth - 1)])
        for p in self.deep_prompts_list_text:
            nn.init.normal_(p, std=0.02)       

    def _init_deep_prompts_vision(self):
        """
        初始化深层视觉prompts
        """
        pass

    def _init_proj(self):
        """
        初始化输入层交互方式
        """
        self.proj = nn.Linear(self.prompt_dim, _VISION_DIM)
        self.proj.half()
    
    def _init_deep_projs(self):
        """
        初始化深层交互方式
        """
        proj = nn.Linear(self.prompt_dim, _VISION_DIM)
        self.deep_projs = _get_clones(proj, self.deep_prompts_depth - 1)

    def _init_tokenized_prompts(self, clip_model):
        """
        初始化离散形式的prompts
        """
        classnames = [name.replace("_", " ") for name in self.classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) #24, 77

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)  #22,77,512

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.prompt_len:, :]) 

        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens


class TrainerX(TrainerX_):
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
