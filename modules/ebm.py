import torch
import torch.nn as nn
import torch.nn.functional as F



class EBM(nn.Module):
    def __init__(self, dim, num_frames, config):
        super(EBM, self).__init__()

        self.embed_dim = dim
        self.config = config

        if self.config.energy_fn == 'mlp':
            self.energy_net = nn.Sequential(nn.Linear(2*self.embed_dim, self.embed_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.embed_dim, 1))
        elif self.config.energy_fn == 'bilinear':
            self.energy_net = nn.Linear(self.embed_dim, self.embed_dim)
        elif self.config.energy_fn == 'cossim':
            self.energy_net = None
        else:
            raise NotImplementedError

        # hyper-params
        self.mcmc_coef_reg = config.mcmc_coef_reg
        self.mcmc_steps = config.mcmc_steps  # number of mcmc sampling
        self.mcmc_step_size = config.mcmc_step_size  # grad coef of mcmc sampling
        self.mcmc_noise = config.mcmc_noise  # noise of mcmc sampling

        # replay buffer
        self.num_frames = num_frames
        self.max_buffer_vol = config.max_buffer_vol
        self.buffer_prob = 0.95
        self.register_buffer('replay_buffer_vid', torch.zeros([self.max_buffer_vol, self.num_frames, self.embed_dim]))
        self.vid_buffer_flag = False
        self.register_buffer('replay_buffer_txt', torch.zeros([self.max_buffer_vol, self.embed_dim]))
        self.txt_buffer_flag = False

        self.non_diag_mask = None

    def energy_pooling(self, energy, dim=-1):
        if self.config.energy_pooling == 'avg':
            energy = energy.mean(dim=dim)  # (bs)
        elif self.config.energy_pooling == 'max':
            energy = energy.max(dim=dim)[0]
        elif self.config.energy_pooling == 'min':
            energy = energy.min(dim=dim)[0]
        else:
            raise NotImplementedError
        return energy

    def energy_diag(self, vid, txt):
        # vid: (bsv, F, hdim)
        # txt: (bst, hdim)
        if self.config.energy_fn == 'mlp':
            bsv, nf, hdim = vid.shape
            bst, _ = txt.shape
            txt_repeat = txt.view(bst, 1, hdim).expand(-1, nf, -1)  # (bst, F, hdim)
            energy = self.energy_net(torch.cat([vid, txt_repeat], dim=-1)).squeeze(-1)  # (bs, F)
        elif self.config.energy_fn in ['bilinear', 'cossim']:
            vid = F.normalize(vid, p=2, dim=-1)
            txt = F.normalize(txt, p=2, dim=-1)
            if self.config.energy_fn == 'bilinear':
                energy = -torch.einsum('ad,afd->af', [self.energy_net(txt), vid])  # (bs, F)
            else:
                energy = -torch.einsum('afd,ad->af', [vid, txt])  # (bs, F)
        else:
            raise NotImplementedError
        energy = self.energy_pooling(energy, dim=-1)
        return energy

    def energy(self, vid, txt):
        # vid: (bsv, F, hdim)
        # txt: (bst, hdim)
        if self.config.energy_fn == 'mlp':
            bsv, nf, hdim = vid.shape
            bst, _ = txt.shape
            vid_repeat = vid.view(1, bsv, nf, hdim).expand(bst, -1, -1, -1)  # (bst, bsv, F, hdim)
            txt_repeat = txt.view(bst, 1, 1, hdim).expand(-1, bsv, nf, -1)   # (bst, bsv, F, hdim)
            energy = self.energy_net(torch.cat([vid_repeat, txt_repeat], dim=-1)).squeeze(-1)  # (bst, bsv, F)
        elif self.config.energy_fn in ['bilinear', 'cossim']:
            vid = F.normalize(vid, p=2, dim=-1)
            txt = F.normalize(txt, p=2, dim=-1)
            if self.config.energy_fn == 'bilinear':
                energy = -torch.einsum('ad,bfd->abf', [self.energy_net(txt), vid])  # (bst, bsv, F)
            else:
                energy = -torch.einsum('bfd,ad->abf', [vid, txt])  # (bst, bsv, F)
        else:
            raise NotImplementedError
        energy = self.energy_pooling(energy, dim=-1)
        return energy

    def loss_compute(self, vid, txt):
        # vid: (bsv, F, hdim)
        # txt: (bst, hdim)
        bsv, nf, hdim = vid.shape
        bst, _ = txt.shape
        assert bsv == bst, "video_feat and text_feat should have same batch size"
        sample_size = bsv

        # energy_pos = self.energy_diag(vid, txt)  # (bs)
        energy = self.energy(vid, txt)  # (bst, bsv)
        energy_pos = energy.diag()  # (bs)
        if self.non_diag_mask is None:
            self.non_diag_mask = ~torch.eye(sample_size, dtype=torch.bool)
        energy_neg = energy[self.non_diag_mask]  # (bs(bs-1))
        loss_ebm1 = self.ebm_loss(energy_pos, energy_neg)

        # (bsv, F, hdim), (bst, hdim)
        vid_sampled, txt_sampled = self.sample(sample_size=sample_size, device=vid.device)
        energy_sampled = self.energy_diag(vid_sampled, txt_sampled)  # (bs)
        loss_ebm2 = self.ebm_loss(energy_pos, energy_sampled)

        loss_ebm = (loss_ebm1 + loss_ebm2) / 2.0
        return loss_ebm

    def ebm_loss(self, energy_pos, energy_neg):
        # energy: (n)
        def _reduction(x):
            return x.mean()
        loss = _reduction(energy_pos) - _reduction(energy_neg)
        loss_reg = _reduction(energy_pos.pow(2)) + _reduction(energy_neg.pow(2))
        return loss + self.mcmc_coef_reg * loss_reg

    def energy_gradient(self, vid, txt):
        self.energy_net.eval()

        # copy data
        vid_i = vid.clone().detach()
        vid_i.requires_grad = True
        txt_i = txt.clone().detach()
        txt_i.requires_grad = True

        # compute gradient
        E_i = self.energy_diag(vid_i, txt_i)  # (n)
        vid_i_grad, txt_i_grad = torch.autograd.grad(E_i.sum(), [vid_i, txt_i], retain_graph=True)

        self.energy_net.train()

        return vid_i_grad, txt_i_grad

    def langevine_dynamics_step(self, vid_old, txt_old):
        # gradient wrt x_old
        grad_energy_vid, grad_energy_txt = self.energy_gradient(vid_old, txt_old)

        # sample eta ~ Normal(0, alpha)
        epsilon_vid = torch.randn_like(grad_energy_vid).to(vid_old.device) * self.mcmc_noise
        epsilon_txt = torch.randn_like(grad_energy_txt).to(txt_old.device) * self.mcmc_noise

        # new sample
        vid_new = vid_old - self.mcmc_step_size * grad_energy_vid + epsilon_vid
        txt_new = txt_old - self.mcmc_step_size * grad_energy_txt + epsilon_txt

        return vid_new.detach(), txt_new.detach()

    def sample(self, sample_size, device):
        grad_enabled = True
        if not torch.is_grad_enabled():
            grad_enabled = False
            torch.set_grad_enabled(True)

        # 1) init sample
        # if self.replay_buffer_vid is None or self.replay_buffer_txt is None:
        if not self.vid_buffer_flag or not self.txt_buffer_flag:
            vid_sample = 2. * torch.rand([sample_size, self.num_frames, self.embed_dim]).to(device) - 1.
            txt_sample = 2. * torch.rand([sample_size, self.embed_dim]).to(device) - 1.
            self.vid_buffer_flag = True
            self.txt_buffer_flag = True
        else:
            replay_sample_idx = torch.randperm(self.replay_buffer_vid.shape[0])[:int(sample_size * self.buffer_prob)]
            vid_sample = torch.cat([self.replay_buffer_vid[replay_sample_idx], 
                                    torch.rand([sample_size - replay_sample_idx.shape[0], self.num_frames, self.embed_dim]).to(device)], dim=0)
            txt_sample = torch.cat([self.replay_buffer_txt[replay_sample_idx], 
                                    torch.rand([sample_size - replay_sample_idx.shape[0], self.embed_dim]).to(device)], dim=0)

        # 2) run Langevine Dynamics sampling
        for _ in range(self.mcmc_steps):
            vid_sample, txt_sample = self.langevine_dynamics_step(vid_sample, txt_sample)
        if self.replay_buffer_vid is None or self.replay_buffer_txt is None:
            self.replay_buffer_vid = vid_sample
            self.replay_buffer_txt = txt_sample
        else:
            self.replay_buffer_vid = torch.cat([vid_sample, self.replay_buffer_vid], dim=0)[:self.max_buffer_vol]
            self.replay_buffer_txt = torch.cat([txt_sample, self.replay_buffer_txt], dim=0)[:self.max_buffer_vol]

        if not grad_enabled:
            torch.set_grad_enabled(False)

        return vid_sample, txt_sample