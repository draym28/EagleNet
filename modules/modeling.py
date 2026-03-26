from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn.functional as F
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn, SigLoss, make_patch_shift
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
from modules.transformer_eaglenet import Transformer as TransformerXPool
from modules.stochastic_module import StochasticText
from modules.gnn import GAT, RGAT
from modules.ebm import EBM

from modules.module_clip import CLIP, convert_weights

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            # This step is to detect whether in train mode or test mode
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break

            # train mode
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        # state_dict["text_prompt_encoder.pos_embedding"] = val[0:3].clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
            # test mode
            else:
                for key, val in state_dict.items():
                    # test mode
                    if  key.find("clip.visual.transformer.resblocks") == 0:
                            num_layer = int(key.split(".")[4])
                            # shift layers 10-11
                            if num_layer >=10 and num_layer < 12:
                                state_dict[key.replace("attn.net.", "attn.")] = val.clone()
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        make_patch_shift(model, video_frame=task_config.max_frames, n_div=7)
        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class EagleNet(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(EagleNet, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            # self.frame_position_embeddings = nn.Embedding(600, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fn = self.task_config.loss_fn
        if self.loss_fn == 'clip':
            self.loss_fct = CrossEn()
        self.apply(self.init_weights)

        if self.loss_fn == 'sig':
            self.loss_fct = SigLoss(self.task_config)

        self.set_dim = 512
        self.num_frames = self.task_config.max_frames
        self.stochastic_prior = self.task_config.stochastic_prior
        self.stochastic_prior_std = self.task_config.stochastic_prior_std

        self.pool_frames = TransformerXPool(self.set_dim, self.task_config.pooling_head, 
                                            self.task_config.pooling_dropout)
        self.stochastic = StochasticText(self.set_dim, self.num_frames, 
                                         self.stochastic_prior, self.stochastic_prior_std)

        if self.task_config.gnn_type == 'gat':
            self.gnn = GAT(self.set_dim, self.set_dim, self.task_config.gnn_num_layers, self.task_config)
        elif self.task_config.gnn_type == 'rgat':
            self.gnn = RGAT(self.set_dim, self.set_dim, self.task_config.gnn_num_layers, self.task_config)
            self.adj_t2t = None
            self.adj_f2f = None
            self.adj_f2t = None
        else:
            raise NotImplementedError

        if self.task_config.framepe:
            self.frame_pe = nn.Parameter(torch.randn([self.num_frames, self.set_dim]))

        self.ebm = EBM(self.set_dim, self.num_frames, self.task_config)
        self.diag_idx = None

        #-------------------------------------------------------------------------------------------------------


    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        # input_ids:      (batch, pairs, M)
        # token_type_ids: (batch, pairs, M)
        # attention_mask: (batch, pairs, M)
        # video           (batch, pairs, F, T, C, H, W)
        # video_mask:     (batch, pairs, F)
        # pairs & T is always 1

        input_ids = input_ids.view(-1, input_ids.shape[-1])                 # (batch*pairs, M)
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])  # (batch*pairs, M)
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])  # (batch*pairs, M)
        video_mask = video_mask.view(-1, video_mask.shape[-1])              # (batch*pairs, F)
        # print('video mask shape in forward: ', video_mask.shape)

        # T x 3 x H x W
        video = torch.as_tensor(video).float()  # (batch, pairs, F, T, C, H, W)
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)  # (batch*pairs*F*T, C, H, W)
        video_frame = bs * ts  # F*T

        # (batch*pairs, hdim), (batch*pairs, F, hdim)
        text_feat, video_feat = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                video, video_mask, shaped=True, video_frame=video_frame)

        def _sim_loss(sim):
            if self.loss_fn == 'clip':
                return (self.loss_fct(sim) + self.loss_fct(sim.t())) / 2
            elif self.loss_fn == 'sig':
                return self.loss_fct(sim)

        if self.training:
            # (bst, bsv), (bsv, bst)
            sim_matrix1, sim_matrix2, eam_loss, eam_support_loss = self.get_max_similarity_logits(
                text_feat, video_feat, attention_mask, video_mask, shaped=True)

            sim_loss_stochastic = _sim_loss(sim_matrix1)
            sim_loss = sim_loss_stochastic
            if sim_matrix2 is not None:
                sim_loss_support = _sim_loss(sim_matrix2)
                sim_loss = sim_loss + sim_loss_support * self.task_config.support_loss_weight

            if eam_loss is not None:
                sim_loss = sim_loss + eam_loss * self.task_config.eam_loss_weight
            if eam_support_loss is not None:
                sim_loss = sim_loss + eam_support_loss * self.task_config.eam_support_loss_weight

            loss = sim_loss

            return loss
        else:
            return None


    def get_max_similarity_logits(self, text_feat, video_feat, text_mask, video_mask, shaped=False):
        # text_feat:  (bst, hdim)
        # video_feat: (bsv, F, hdim)
        # text_mask:  (bst, M)
        # video_mask: (bsv, F)

        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if self.task_config.n_gpu > 1:
            if self.training and torch.cuda.is_available():  # batch merge here
                text_feat = allgather(text_feat.contiguous(), self.task_config)
                video_feat = allgather(video_feat.contiguous(), self.task_config)
                video_mask = allgather(video_mask.contiguous(), self.task_config)
                torch.distributed.barrier()  # force sync

        bst, hdim = text_feat.shape
        bsv, nf, _ = video_feat.shape
        text_feat_repeat = text_feat.view(bst, 1, hdim).repeat(1, bsv, 1).reshape(bst*bsv, hdim)
        video_feat_repeat = video_feat.view(1, bsv, nf, hdim).repeat(bst, 1, 1, 1).reshape(bst*bsv, nf, hdim)

        ns = self.task_config.stochasic_trials
        # (ns, bst*bsv, hdim), (bst*bsv, hdim), (bst*bsv, hdim)
        text_feat_stochastic, _, text_log_var = self.stochastic.stochastic_ntimes(
            text_feat_repeat, video_feat_repeat, ns)

        if self.task_config.gnn_type == 'rgat':
            if self.adj_t2t is None:  # initialize adj
                self.adj_t2t = torch.zeros([nf+1+ns, nf+1+ns])
                self.adj_t2t[nf:, nf:] = 1
                self.adj_f2f = torch.zeros([nf+1+ns, nf+1+ns])
                self.adj_f2f[:nf, :nf] = 1
                self.adj_f2t = torch.zeros([nf+1+ns, nf+1+ns])
                self.adj_f2t[nf:, :nf] = 1
                self.adj_f2t[:nf, nf:] = 1

        # select the best text_sto
        if self.task_config.framepe:
            video_feat_repeat = video_feat_repeat + self.frame_pe.unsqueeze(0)  # (bst*bsv, F, hdim)
        nodes = torch.cat([video_feat_repeat,                       # (bst*bsv, F, hdim)
                           text_feat_repeat.unsqueeze(1),           # (bst*bsv, 1, hdim)
                           text_feat_stochastic.permute(1, 0, 2)],  # (bst*bsv, ns, hdim)
                          dim=1)  # (bst*bsv, F+1+ns, hdim)

        if self.task_config.gnn_type == 'rgat':
            adjs_t2t = self.adj_t2t.to(nodes.device).view(1, nf+1+ns, nf+1+ns).repeat(nodes.shape[0], 1, 1)
            adjs_f2f = self.adj_f2f.to(nodes.device).view(1, nf+1+ns, nf+1+ns).repeat(nodes.shape[0], 1, 1)
            adjs_f2t = self.adj_f2t.to(nodes.device).view(1, nf+1+ns, nf+1+ns).repeat(nodes.shape[0], 1, 1)

        if self.task_config.gnn_type == 'gat':
            # (bst*bsv, F+1+ns, hdim), [2 * (bst*bsv, nh, F+1+ns, F+1+ns)]
            _, atts_tuple = self.gnn(nodes, return_atts=True)
            eij_f2t = atts_tuple[0]                        # (bst*bsv, nh, F+1+ns, F+1+ns), eij
            atts_video_text = eij_f2t[:, :, :nf, nf:]      # (bst*bsv, nh, F, 1+ns)
            atts_video_text = atts_video_text.mean(dim=1)  # (bst*bsv, F, 1+ns)
        elif self.task_config.gnn_type == 'rgat':
            # (bst*bsv, F+1+ns, hdim), [n_rel * (bst*bsv, nh, F+1+ns, F+1+ns)]
            _, atts_tuple = self.gnn(nodes, [adjs_t2t, adjs_f2f, adjs_f2t], return_atts=True)
            eij_f2t = atts_tuple[-1]                       # (bst*bsv, nh, F+1+ns, F+1+ns), eij
            atts_video_text = eij_f2t[:, :, :nf, nf:]      # (bst*bsv, nh, F, 1+ns)
            atts_video_text = atts_video_text.mean(dim=1)  # (bst*bsv, F, 1+ns)
        else:
            raise NotImplementedError
        atts_video_text = atts_video_text.mean(dim=1)  # (bst*bsv, 1+ns)
        atts_video_text = torch.softmax(atts_video_text, dim=-1)
        text_feat_stochastic = torch.einsum('at,atd->ad', 
                                            [atts_video_text, nodes[:, nf:]])  # (bst*bsv, hdim)

        # reshape
        text_feat_stochastic = text_feat_stochastic.view(bst, bsv, hdim)
        text_log_var = text_log_var.view(bst, bsv, hdim)

        video_feat_pooled = self.pool_frames(text_feat_stochastic, video_feat)  # (bsv, bst, hdim)

        # stochastic loss
        text_feat_stochastic = F.normalize(text_feat_stochastic, p=2, dim=-1)  # (bst, bsv, hdim)
        video_feat_pooled = F.normalize(video_feat_pooled, p=2, dim=-1)        # (bsv, bst, hdim)
        retrieve_logits = torch.einsum('abd,bad->ab', [text_feat_stochastic, video_feat_pooled])

        if self.training:
            # support loss
            if not self.task_config.support_loss_weight == 0:
                video_feat_pooled_avg = video_feat_pooled.mean(dim=1)  # (bsv, hdim)
                pointer = video_feat_pooled_avg.view(1, bsv, hdim) - text_feat.view(bst, 1, hdim)  # (bst, bsv, hdim)
                text_support = F.normalize(pointer, p=2, dim=-1) * torch.exp(text_log_var) + text_feat.view(bst, 1, hdim)  # (bst, bsv, hdim)
                text_support = F.normalize(text_support, p=2, dim=-1)
                retrieve_logits_2 = torch.einsum('abd,bad->ab', [text_support, video_feat_pooled])  # (bst, bsv)
            else:
                retrieve_logits_2 = None

            # EBM loss
            if not self.task_config.eam_loss_weight == 0:
                eam_loss = self.ebm.loss_compute(video_feat, text_feat)
            else:
                eam_loss = None

            # EBM support loss
            if not self.task_config.support_loss_weight == 0 and not self.task_config.eam_support_loss_weight == 0:
                if self.diag_idx is None:
                    self.diag_idx = torch.arange(text_support.shape[0]).to(text_support.device)
                eam_support_loss = self.ebm.loss_compute(video_feat, text_support[self.diag_idx, self.diag_idx, :])
            else:
                eam_support_loss = None

            if self.loss_fn == 'clip':
                logit_scale = self.clip.logit_scale.exp()
                retrieve_logits = logit_scale * retrieve_logits
                if retrieve_logits_2 is not None:
                    retrieve_logits_2 = logit_scale * retrieve_logits_2
            return retrieve_logits, retrieve_logits_2, eam_loss, eam_support_loss

        else:
            return retrieve_logits, retrieve_logits.t()


    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        # input_ids:      (batch*pairs, M)
        # token_type_ids: (batch*pairs, M)
        # attention_mask: (batch*pairs, M)

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)  # batch*pairs
        sequence_hidden = self.clip.encode_text(input_ids).float()  # (batch*pairs, hdim)

        # (batch*pairs, hdim)
        return sequence_hidden


    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        # video:      (batch*pairs*F*T, C, H, W)
        # video_mask: (batch*pairs, F)

        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)  # batch*pairs

        video_feat = self.clip.encode_image(video).float()  # (batch*pairs*F, hdim)
        video_feat = video_feat.reshape(bs_pair, video_mask.shape[1], -1)  # (batch*pairs, F, hdim)

        # (batch*pairs, F, hdim)
        return video_feat


    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        # input_ids:      (batch*pairs, M)
        # token_type_ids: (batch*pairs, M)
        # attention_mask: (batch*pairs, M)
        # video:          (batch*pairs*F*T, C, H, W)
        # video_mask:     (batch*pairs, F)

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        # (batch*pairs, hdim)
        text_feat = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)

        # (batch*pairs, F, hdim)
        video_feat = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        # (batch*pairs, hdim), (batch*pairs, F, hdim)
        return text_feat, video_feat
