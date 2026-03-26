from math import isnan
import torch
import torch.nn as nn


class LinearCosRadius(nn.Module):
    '''
    Define the radius (R) as the linear function of the cos-similarity (t, v)
    '''
    def __init__(self, embed_dim, num_frames):
        super(LinearCosRadius, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.linear_proj = nn.Linear(self.num_frames, self.embed_dim)
        self.learnable_scalar = nn.Parameter(torch.Tensor(1))

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        # normalization
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (bst, hdim)
        video_tmp = video_embeds / video_embeds.norm(dim=-1, keepdim=True)  # (bsv, #F, hdim)
        if torch.isnan(video_tmp).sum() == 0:
            video_embeds = video_tmp

        # sim computation
        text_embeds = text_embeds.unsqueeze(1).repeat(1, self.num_frames, 1)  # (bst, #F, hdim)
        sims = torch.matmul(text_embeds, video_embeds.permute(0,2,1))  # (bs, #Ft, #Fv)
        sims = torch.mean(sims, dim=1)  # (bs, #Fv), S

        # linear proj
        sims_out = self.linear_proj(sims)  # (bs, hdim), SW

        return sims_out


class StochasticText(nn.Module):
    def __init__(self, embed_dim, num_frames, stochastic_prior, stochastic_prior_std):
        super(StochasticText, self).__init__()

        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.stochastic_prior = stochastic_prior
        self.stochastic_prior_std = stochastic_prior_std

        self.std_branch = LinearCosRadius(embed_dim, num_frames)

    def forward(self, text_features, video_features):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_texts x embed_dim
        """
        # @WJM: re-parameterization for text (independent of the text pool video)
        text_mean = text_features  # (bst, hdim)

        # radius
        log_var = self.std_branch(text_features, video_features)  # (bs, hdim), SW
        text_std = torch.exp(log_var) # log var -> var, R=exp(SW)

        # randomness
        if self.stochastic_prior == 'uniform01':
            sigma = torch.rand_like(text_features)  # (bs, hdim)
        elif self.stochastic_prior == 'normal':
            sigma = torch.normal(mean=0., std=self.stochastic_prior_std, size=text_features.shape).to(text_std.device)
        else:
            raise NotImplementedError

        # re-parameterization
        text_features = text_mean + sigma * text_std  # (bs, hdim)

        return text_features, text_mean, log_var

    def stochastic_ntimes(self, text_features, video_features, num_stochastic):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_stochastic x num_texts x embed_dim
        """
        # @WJM: re-parameterization for text (independent of the text pool video)
        text_mean = text_features  # (bst, hdim)

        # radius
        log_var = self.std_branch(text_features, video_features)  # (bs, hdim), SW
        text_std = torch.exp(log_var) # log var -> var, R=exp(SW)

        bst, hdim = text_features.shape
        device = text_features.device

        # randomness
        if self.stochastic_prior == 'uniform01':
            sigma = torch.rand(num_stochastic, bst, hdim).to(device)  # (ns, bs, hdim)
        elif self.stochastic_prior == 'normal':
            sigma = torch.normal(mean=0., std=self.stochastic_prior_std, 
                                 size=[num_stochastic, bst, hdim]).to(device)
        else:
            raise NotImplementedError

        # re-parameterization
        text_features = text_mean.view(1, bst, hdim) + sigma * text_std.view(1, bst, hdim)  # (ns, bs, hdim)

        # (ns, bs, hdim), (bs, hdim), (bs, hdim)
        return text_features, text_mean, log_var
