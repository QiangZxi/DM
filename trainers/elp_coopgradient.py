# CoOp w/ DePT
import os
import os.path as osp
import torch
import torch.nn.functional as F
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_lr_scheduler
from plotnine import *
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from sklearn.decomposition import PCA
import numpy as np
from clip import clip
from torch.nn.modules.loss import _Loss

from .coop import CoOp, load_clip_to_cpu
from .coop import CustomCLIP as CustomCLIP_
from .elp_maple import FiLM
from .optim import build_optimizer
from .gradient_projection import get_featuremat
from .base import TrainerX


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.dataset = cfg.DATASET.NAME
        self.lp_cfg = cfg.TRAINER.LINEAR_PROBE
        self.film_cfg = cfg.TRAINER.FILM
        
        clip_dim = clip_model.text_projection.size(1)
        
        film_cfg = self.film_cfg

        if film_cfg.LINEAR_PROBE:
            # cwT module
            self.film_lp_img = FiLM(clip_dim)
            self.film_lp_text = FiLM(clip_dim)
        
        # for base to new, base classes will be 'base'
        # for cross dataset, classes from ImageNet will be 'base'
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            assert self.lp_cfg.TYPE in ['similarity', 'linear']

            # linear classifier
            if self.lp_cfg.TYPE == 'similarity':
                self.linear_probe_proj = nn.Identity()
            elif self.lp_cfg.TYPE == 'linear':
                self.linear_probe_proj = nn.Linear(clip_dim, len(classnames)).type(self.dtype)
        else:
            self.linear_probe_proj = nn.Identity()
        
    def forward(self, img, labels=None):
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            out = self._forward_base(img, labels)
            return out
        else:
            return self._forward_new(img)

    def _forward_base(self, img, labels=None):
        """ forward function for base classes """
        text_feats, img_feats = self._forward_feats(img)
        lp_weight = self.lp_cfg.WEIGHT

        # forward similartiy and linear logits
        logits = self._forward_logits_similarity(text_feats, img_feats)
        logits_lp, labels_lp, logits_out= self._forward_logits_linear_probe(text_feats, img_feats, labels)

        #logits, logits_lp, (1 - lp_weight) * logits + lp_weight * logits_lp
        if self.prompt_learner.training:
            # while training, return loss of both logits
            logits_out = logits
            return self._loss(logits, labels, logits_lp, labels_lp), logits_out
            #return 
        
        if not self.lp_cfg.TEST_TIME_FUSION:
            return logits_lp

        # while inference, fusion both logits and return
        
        logits = (1 - lp_weight) * logits + lp_weight * logits_lp
        return logits
    
    def _forward_new(self, img):
        """ forward function for new classes """
        assert not self.prompt_learner.training
        
        # for new classes, only forward similarity logits
        text_feats, img_feats = self._forward_feats(img)
        logits = self._forward_logits_similarity(text_feats, img_feats)
        return logits
    
    def _forward_feats(self, img):
        prompts = self.prompt_learner()

        tokenized_prompts = self.tokenized_prompts
        text_feats = self.text_encoder(prompts, tokenized_prompts)
        img_feats = self.image_encoder(img.type(self.dtype))

        return text_feats, img_feats
    
    def _forward_logits_similarity(self, text_feats, img_feats):
        # normalize and calcute cosine similarity
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.t()
        return logits
    
    def _forward_logits_linear_probe(self, text_feats, img_feats, labels):
        # cwT module
        if self.film_cfg.LINEAR_PROBE:
            text_feats = self.film_lp_text(text_feats)
            img_feats = self.film_lp_img(img_feats)

        # while new head is similarity head, use similarity forward function
        if self.lp_cfg.TYPE == 'similarity':
            return self._forward_logits_similarity(text_feats, img_feats), labels
 
        if labels is None:
            # while inference, forward image features only
            all_feats = img_feats
            all_labels = labels
        else:
            # while training, image features and text features will be concated to train classifier
            text_feats = text_feats[labels]
            all_feats = torch.cat([text_feats, img_feats])
            all_labels = torch.cat([labels, labels])
        all_logits = self.linear_probe_proj(all_feats)
        logit_out = (all_logits[0] + all_logits[1])/2
        return all_logits, all_labels, logit_out

    def _loss(self, logits, labels, logits_lp, labels_lp):
        # calculate similarity loss and linear loss
        loss_cls = F.cross_entropy(logits, labels)
        loss_cls_lp = F.cross_entropy(logits_lp, labels_lp)

        lp_weight = self.lp_cfg.WEIGHT
        loss = (1 - lp_weight) * loss_cls + lp_weight * loss_cls_lp
        return loss

class CLIP_(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model.float()

        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits
    
class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss

@TRAINER_REGISTRY.register()
class ExtrasLinearProbeCoOp_g(CoOp):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                loss, logits = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss, logits = model(image, label)
            logits_clip = self.clip_model_(image)

            xe_loss, kl_loss = self.ProGradLoss(logits, logits_clip, label)
            optim.zero_grad()
            saved_grads = {}
            kl_loss.backward(retain_graph=True)
            for k, (m, params) in enumerate(self.model.named_parameters()):
                if "prompt_learner" in m:
                    """ for name_ in names_to_update:
                    if name_ not in m and params.grad is not None: """
                    saved_grads[m] = params.grad.data.clone()
            self.optim.zero_grad()
            xe_loss.backward()
            for k, (m, params) in enumerate(self.model.named_parameters()):
                if "prompt_learner" in m:
                    temp = np.dot(params.grad.data.detach().flatten().cpu().numpy(), saved_grads[m].detach().flatten().cpu().numpy())
                    if temp < 0:
                        params.grad.data = params.grad.data - temp/np.linalg.norm(saved_grads[m].cpu().numpy(), ord=2)*saved_grads[m]
            """ if "prompt_learner" in m:
                    temp = np.einsum('ij,ij->j', params.grad.data.detach().cpu().numpy(), saved_grads[m].detach().cpu().numpy())
                    for i in range(len(temp)):
                        if temp[i] < 0:
                            params.grad.data = params.grad.data - 0.1*temp[i]/np.linalg.norm(saved_grads[m][i].cpu().numpy(), ord=2)*saved_grads[m] """
            #loss.backward()
            """ for k, (m, params) in enumerate(model.named_parameters()):
                        if "image_encoder" or "text_encoder" in m:
                            params.grad.data = params.grad.data - torch.matmul(params.grad.data, feature_mat) """
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

        
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        self.clip_model_ = CLIP_(cfg, classnames, clip_model)
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.ProGradLoss = ProGradLoss(T=0.5)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = cfg.TRAINER.NAMES_TO_UPDATE

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
                
            param.requires_grad_(update)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print(f"Parameters to be updated: {list(sorted(enabled))}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim, infos = build_optimizer(self.model, cfg.OPTIM)

        if infos is not None:
            print('Learning rate of parameters:')
            for info in infos:
                print('lr: {}, layers: {}'.format(info['lr'], info['layers']))
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("PromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            if epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)

            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            # for some dataset in domain generalization, number of target classes is different from number of source classes
            # thus a mapping must be created to preserve the required class weights
            if self.cfg.DATASET.NAME in ['ImageNetA', 'ImageNetR']:
                from datasets.imagenet import ImageNet
                from dassl.utils import listdir_nohidden

                # read classes from source dataset
                dataset = self.dm.dataset
                text_file = osp.join(dataset.dataset_dir, "classnames.txt")
                all_folders = ImageNet.read_classnames(text_file).keys()

                # read classes from target dataset
                TO_BE_IGNORED = ["README.txt"]
                folders = listdir_nohidden(dataset.image_dir, sort=True)
                folders = [f for f in folders if f not in TO_BE_IGNORED]

                # find that which class from target dataset is in source dataset
                is_reserves = [f in folders for f in all_folders]

                # only reserve required class weights
                print(f'State dict is CLIPPED to match the shape of target dataset {self.cfg.DATASET.NAME}!')
                state_dict['linear_probe_proj.weight'] = state_dict['linear_probe_proj.weight'][is_reserves]
                state_dict['linear_probe_proj.bias'] = state_dict['linear_probe_proj.bias'][is_reserves]
                
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
