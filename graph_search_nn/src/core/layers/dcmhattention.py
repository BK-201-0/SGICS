import torch
import torch.nn as nn
from torch import Tensor
from ..utils.constants import INF
import math, copy
import torch.nn.functional as F
from typing import Optional,Tuple,List

class CrossHeadProjection(nn.Module):

    def __init__(self, mode, num_heads=16, num_groups=1, dtype=torch.float32, use_sw=False):
        super().__init__()
        self.mode = mode
        self.use_sw = use_sw
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_heads_per_group = self.num_heads // self.num_groups
        if self.use_sw:
            self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group, dtype=dtype))
        else:
            self.register_buffer('w', torch.eye(self.num_heads_per_group, dtype=dtype).expand(self.num_groups, self.num_heads_per_group, self.num_heads_per_group))

    def forward(self, inputs,
            dws:Optional[Tuple[Tensor,Tensor, Tensor,Tensor, Tensor,Tensor]]=None,
            query_vec=None, key_vec=None,
            proj_w:Optional[Tensor]=None,
            fast_infer=True):
        if proj_w is not None:
            ret = torch.einsum('BNTS,BSNM->BMTS', inputs, proj_w)
        else:
            assert dws is not None
            qw1, qw2, kw1, kw2, qdd, kdd = dws
            inputs = inputs.unsqueeze(1) #BNTS->BGNTS
            # apply sw
            ret = torch.einsum('BGMTS,GMN->BGNTS', inputs, self.w) if self.use_sw else inputs
            if fast_infer:
                inputs_label = 'BGMTS'
                hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I') # BGITS
                # apply qw and kw
                for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
                    dw_label = f'B{sym}G{hidden_sym}M'  # w1: BTGIM, dw_label:BTGIM
                    dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
                    eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'
                    eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'
                    for i in range(dynamic_hidden_dim):
                        hidden = torch.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :]) # BGMTS,BTG(I)M->BGTS
                        out = torch.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :]) #  'BG(I)TS,BTG(I)M->BGMTS'
                        ret = ret + out
                # apply qdd and kdd
                for sym, dd in zip(['T', 'S'], [qdd, kdd]):
                    dd_label = f'B{sym}GM'
                    dout = torch.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd) # BGMTS,B(T/S)GM->BGMTS
                    ret = ret + dout
            else:
                # apply qw and kw (BTGIN)
                x_inter = torch.einsum('BGNTS, BTGIN->BGTSI', inputs, qw1)
                qw_out = torch.einsum('BGTSI, BTGIN->BGNTS', x_inter, qw2)
                ret = ret + qw_out
                x_inter = torch.einsum('BGNTS, BSGIN->BGTSI', inputs, kw1)
                kw_out = torch.einsum('BGTSI, BSGIN->BGNTS', x_inter, kw2)
                ret = ret + kw_out

                # apply qdd(BTGN) and kdd(BSGN)
                ret = ret + torch.einsum('BGNTS, BTGN->BGNTS', inputs, qdd)
                ret = ret + torch.einsum('BGNTS, BSGN->BGNTS', inputs, kdd)
            ret = ret.squeeze(1) # BGNTS->BNTS
        return ret

class RMSnormNoscale(nn.Module):

    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs

class RMSnorm(nn.Module):

    def __init__(self, hid_dim=128, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.scale = nn.parameter.Parameter(data=torch.ones(self.hid_dim))

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs

class DynamicWeightProjection(nn.Module):

    def __init__(self, num_heads=32, num_groups=1, residual=True, query_input_dim=4096, dynamic_squeeze_ratio=16,
                 dynamic_w_hidden_dim=128, dtype=torch.float32, use_sw=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.query_input_dim = query_input_dim
        self.dynamic_squeeze_ratio = dynamic_squeeze_ratio
        self.dynamic_w_hidden_dim = dynamic_w_hidden_dim
        self.dw_hidden_activation = nn.GELU()
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.dw_activation = nn.Tanh()
        self.dw1_norm = RMSnormNoscale(dim=-1)
        self.use_sw = use_sw
        self.pre_proj = CrossHeadProjection('pre', num_heads=self.num_heads, use_sw=use_sw)
        self.post_proj = CrossHeadProjection('post', num_heads=self.num_heads, use_sw=use_sw)

        dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.dw1 = nn.parameter.Parameter(
            torch.zeros(self.query_input_dim, self.num_groups, 4, self.dynamic_w_hidden_dim,
                        dtype=dtype))  # (4096, 1, 4, 128)
        G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
        I = dynamic_hidden_dim * 2
        self.qkw = nn.parameter.Parameter(torch.zeros([G, 4, K, I, M], dtype=dtype))  # (1, 4, 128, 4, 32)
        self.dd = nn.parameter.Parameter(
            torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * 4,
                        dtype=dtype))  # (4096, 1, 128)

        self.merge_weights()

    def merge_weights(self):
        self.dw_m = nn.parameter.Parameter(
            torch.cat([self.dw1.reshape(self.query_input_dim, -1), self.dd.squeeze(1)], dim=-1)).to(
            self.dw1.device)  # E,(4*K + K)  K=2*N*I
        self.qkw_m = nn.parameter.Parameter(
            self.qkw.permute(0, 1, 2, 3, 4).reshape(4, self.dynamic_w_hidden_dim, -1)).to(self.dw1.device)  # (4,K,I*M)
        if self.use_sw:
            self.sw = nn.parameter.Parameter(
                torch.stack([self.pre_proj.w, self.post_proj.w]).squeeze(1) + torch.eye(self.num_heads)).to(
                self.dw1.device)  # (2,N,N) sw + identity matrix
        else:
            self.sw = (torch.eye(self.num_heads).expand(2, self.num_heads, self.num_heads)).to(
                self.dw1.device)  # identity matrix (2,N,N)

    def forward(self, query_vec, KW: Optional[torch.Tensor] = None, gen_cache: Optional[bool] = True):
        dw_hidden = torch.einsum('BTD,DGCK->BTGCK', query_vec, self.dw1)  # C=4 [pre,post]*[query,key]
        dw_hidden = self.dw_hidden_activation(dw_hidden)  # BTGCK
        w1, w2 = torch.split(torch.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), self.qkw.shape[-2] // 2,
                             dim=-2)  # BTGC(2I)M -> [BTGCIM] * 2
        w1 = self.dw1_norm(w1)  # BTGCIM
        pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, dim=3)  # BTG4IM->[BTGIM]*4
        pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, dim=3)
        dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd)  # BTG(4M)
        dd = self.dw_activation(dd)
        pre_qdd, pre_kdd, post_qdd, post_kdd = torch.split(dd, dd.shape[-1] // 4, dim=-1)  # BTG(4N)->[BTGN]*4
        pre_dw_args = (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)
        post_dw_args = (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
        if gen_cache:  # generate KW cache
            pre_kw = torch.einsum('BSGIM, BSGIN->BSMN', pre_kw1, pre_kw2) + torch.diag_embed(
                pre_kdd.squeeze(2))  # merge kw and kdd
            post_kw = torch.einsum('BSGIM, BSGIN->BSMN', post_kw1, post_kw2) + torch.diag_embed(post_kdd.squeeze(2))
            KW = torch.stack((pre_kw, post_kw), dim=-3)  # BSMN,BSMN->BS2MN
        return pre_dw_args, post_dw_args, KW

class DCMHAttention(nn.Module):
    def __init__(self, batch_size, n_head, dim, device, q_chunk_size=64, window_size=128, lidx=0, is_training=True, use_dcmha=True, use_qk_norm=False, query_wise=False, window_type: Optional[str] = None, use_sw=False):
        super().__init__()
        assert dim % n_head == 0
        n_local_heads = n_head
        head_dim = dim // n_head
        total_head_dim = (n_head + 2 * n_local_heads) * head_dim
        # key, query, value projections for all heads, but in a batch
        self.lidx = lidx
        self.wqkv = nn.Linear(dim, total_head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None
        
        # self.freqs_cis = precompute_freqs_cis(batch_size, head_dim,1000).to(device)
        
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_local_heads = n_local_heads
        self.is_training = is_training
        self.dim = dim
        self.use_dcmha = use_dcmha
        self.scale_factor = 1 / math.sqrt(self.head_dim)
        self.q_chunk_size = q_chunk_size
        self.use_sw = use_sw
        self.dyn_w_proj = DynamicWeightProjection(num_heads=self.n_head, query_input_dim=dim,
                                                  dynamic_squeeze_ratio=self.n_head // 2,
                                                  dynamic_w_hidden_dim=self.n_head * 4, use_sw=use_sw)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSnorm(hid_dim=self.head_dim)
            self.k_norm = RMSnorm(hid_dim=self.head_dim)

        self.window_types = {
            "LG": [256, None],
            "LGLL": [256, None, 256, 256],
            "LGL6": [256, None, 256, 256, 256, 256, 256, 256],
        }

        self.query_wise = query_wise
        if window_type is None:  # LG
            self.window_size = None if self.lidx % 2 == 1 else window_size
        else:
            window_l = self.window_types[window_type]
            self.window_size = window_l[self.lidx % len(window_l)]

        if not self.is_training:
            self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def _generate_fast(self, x, input_pos, q, k, v, k_mask):
        B, T, D = x.shape
        N, I = self.n_head, self.dyn_w_proj.dynamic_hidden_dim  # 32, 2
        dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2 * 2 * N * (2 * I), 2 * 2 * N * 1],
                                                         -1)  # BTD, D(4K+4N) -> BT(4K+4N) -> BT(4K), BT(4N)
        dw_hidden = dw_hidden.view((B, T, 4, -1, 1))  # BT(4K) -> BT4K1
        dw = (self.dyn_w_proj.dw_hidden_activation(dw_hidden) * self.dyn_w_proj.qkw_m).sum(
            -2)  # gelu, BT4K1, 4K(IM)->BT4K(IM)->BT4(IM)
        w1, w2 = dw.view((B, T, 2, 2, -1, N)).split(I, -2)  # BT4(IM)->BT{pre/post}{q/k}IM->[BT22IM] * 2
        w1 = self.dyn_w_proj.dw1_norm(w1)  # BT22IN
        qkdd = self.dyn_w_proj.dw_activation(dd.view((B, T, 2, 2, N)))  # BT2{2}N1->BT2{2}N tanh
        qkw = torch.einsum('BTKJIN,BTKJIM->BTKJNM', w1, w2) + torch.diag_embed(qkdd)  # j=k=2, BT2{2}NM q/k, pre/post
        if self.query_wise:  # TODO: do not generate kw and kdd
            qw, _ = qkw.unbind(3)  # BS2NM
            kw_new = None
            qw = qw + self.dyn_w_proj.sw
        else:
            qw, kw_new = qkw.unbind(3)  # BS{pre/post}{q/k}NM -> BS{pre/post}NM * 2
            kw_new = kw_new + self.dyn_w_proj.sw  # BS2NM + 2NM-> BS2NM
        if self.kv_cache is not None:
            k, v, kw_out = self.kv_cache.update(input_pos, k, v, kw_val=kw_new)  # BNT2M
        logits = q @ k.transpose(-2, -1) * self.scale_factor
        if self.query_wise:
            w = qw  # B12NM
        else:
            w = qw + kw_out  # B12NM,BS2NM -> BS2NM
        wl, w = w.permute(0, 2, 3, 4, 1).unbind(1)  # BS2NM->B2NMS->[BNMS]*2
        logits = (logits * wl).sum(1).unsqueeze(2)  # BN1S, BNMS -> BNMS-> BMS-> BM1S
        min_value = torch.finfo(torch.float32).min
        logits = torch.where(k_mask, logits, min_value)
        probs = logits.softmax(-1)
        probs = (probs * w).sum(1).unsqueeze(2)
        y = probs @ v
        return y

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None, fast_infer=True,
                gen_mask=None) -> Tensor:
        print(f"Shape of mask: {mask.shape}")
        print(f"Shape of x: {x.shape}")

        bsz, seqlen, _ = x.shape
        # freqs_cis = self.freqs_cis
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)  # BSND
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # q = apply_rotary_emb(q, freqs_cis)
        # k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))  # BNSD

        if self.is_training:
            N, D, I = self.n_head, self.head_dim, self.dyn_w_proj.dynamic_hidden_dim;  # 6.7B
            B, T, E = x.shape
            if self.use_dcmha:
                project_logits = True
                project_probs = True
                if project_probs:
                    dw_hidden, dd = (x @ self.dyn_w_proj.dw_m).split([2 * 2 * N * (2 * I), 2 * 2 * N * 1], -1)
                    dw_hidden = self.dyn_w_proj.dw_hidden_activation(dw_hidden)
                    dw_hidden = dw_hidden.view(dw_hidden.shape[:2] + (4, -1))  # B T (4 K) -> B T 4 K  # reshape
                    dw = torch.einsum('B T C K, C K D -> B T C D', dw_hidden,
                                      self.dyn_w_proj.qkw_m)  # BT4K,4K(MI)->BT4(MI)
                    shape = (B, T, 2 * 2, -1, N)  # if project_logits else (B,T,2,N,-1)  # BT(pre/post)(q/k)IN
                    w1, w2 = dw.view(shape).split(I, -2)
                    w1 = self.dyn_w_proj.dw1_norm(w1)  # BT22IN
                    if self.use_sw:
                        pre_sw, post_sw = self.dyn_w_proj.sw.unbind(0)
                    else:
                        pre_sw, post_sw = None, None
                    pre_qw1, pre_kw1, post_qw1, post_kw1 = w1.unbind(2)  # BT(2{*2})IN->[BTIN]*4
                    pre_qw2, pre_kw2, post_qw2, post_kw2 = w2.unbind(2)
                    qkdd = F.tanh(dd).squeeze(-1).view(shape[:-2] + (N,))  # BT(2{*2})N1->BT(2{*2})N
                    pre_qdd, pre_kdd, post_qdd, post_kdd = qkdd.unbind(2)  # BT(2{*2})N->[BTN]*4

                y = torch.zeros(B, N, T, D).to(q.device, dtype=torch.float32)
                for i in range(T // self.q_chunk_size):
                    start, stop = i * self.q_chunk_size, (i + 1) * self.q_chunk_size
                    kv_start = max(0, stop - self.q_chunk_size - self.window_size)
                    _q = q[:, :, start: stop, :]
                    _k, _v = k[:, :, kv_start: stop, :], v[:, :, kv_start: stop, :]
                    _atten_mask = mask[:, :, start: stop]
                    _pre_proj_dw_args = slice_dw(pre_sw, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd, start,
                                                 stop, kv_start) \
                        if project_logits else None
                    _post_proj_dw_args = slice_dw(post_sw, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd,
                                                  start, stop, kv_start) \
                        if project_probs else None
                    _o = _atten_context(_q, _k, _v, _atten_mask, _pre_proj_dw_args, _post_proj_dw_args)
                    y[:, :, start:stop] = _o
            else:
                y = torch.zeros(B, N, T, D).to(q.device, dtype=torch.float32)
                for i in range(T // self.q_chunk_size):
                    start, stop = i * self.q_chunk_size, (i + 1) * self.q_chunk_size
                    kv_start = max(0, stop - self.q_chunk_size - self.window_size)
                    _q = q[:, :, start: stop, :]
                    _k, _v = k[:, :, kv_start: stop, :], v[:, :, kv_start: stop, :]
                    _atten_mask = mask[:, :, start: stop, kv_start: stop]
                    _pre_proj_dw_args, _post_proj_dw_args = None, None
                    _o = _atten_context(_q, _k, _v, _atten_mask, _pre_proj_dw_args, _post_proj_dw_args)
                    y[:, :, start:stop] = _o
        else:  # inference
            if seqlen == 1:  # one-token generation
                k_mask = mask if self.window_size is None else gen_mask[:, :, :, :self.kv_cache.seq_length]
                if fast_infer:
                    y = self._generate_fast(x, input_pos, q, k, v, k_mask)
                else:
                    assert not self.query_wise
                    # generate dw from hidden_state
                    pre_proj_dw_args, post_proj_dw_args, kw_new = self.dyn_w_proj(x, gen_cache=True)

                    # update kvkw cache
                    kw_new = kw_new + self.dyn_w_proj.sw  # absorb residual or sw into kw cache
                    if self.kv_cache is not None:
                        k, v, kw_out = self.kv_cache.update(input_pos, k, v, kw_val=kw_new)  # BNSD, BNSD, BS2NN

                    logits = q @ k.transpose(-2, -1) * self.scale_factor
                    # merge pre_w and apply it
                    pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
                    pre_qw = torch.einsum('BTGIN, BTGIM->BTNM', pre_qw1, pre_qw2) + torch.diag_embed(pre_qdd.squeeze(2))
                    pre_w = pre_qw + kw_out[:, :, 0]  # B1NM, BSNM -> BSNM
                    logits = self.dyn_w_proj.pre_proj(logits, proj_w=pre_w.squeeze(1))

                    logits = torch.where(k_mask, logits, torch.finfo(torch.float32).min)
                    probs = logits.softmax(-1)

                    # merge post_w and apply it
                    post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
                    post_qw = torch.einsum('BTGIN, BTGIM->BTNM', post_qw1, post_qw2) + torch.diag_embed(
                        post_qdd.squeeze(2))
                    post_w = post_qw + kw_out[:, :, 1]
                    probs = self.dyn_w_proj.post_proj(probs, proj_w=post_w.squeeze(1))

                    y = probs @ v
            else:  # prefill
                k_mask = mask[:, :, :, :k.shape[-2]]
                pre_proj_dw_args, post_proj_dw_args, kw_new = self.dyn_w_proj(x, gen_cache=True)
                kw_new = kw_new + self.dyn_w_proj.sw  # absorb residual or sw into kw cache
                if self.kv_cache is not None:
                    self.kv_cache.update(input_pos, k, v, kw_val=kw_new)  # BNSD, BNSD, BS2NN
                logits = q @ k.transpose(-2, -1) * self.scale_factor
                logits = self.dyn_w_proj.pre_proj(logits, dws=pre_proj_dw_args, query_vec=x, key_vec=x,
                                                  fast_infer=True)  # XD BN1S
                logits = torch.where(k_mask, logits, torch.finfo(torch.float32).min)
                probs = logits.softmax(-1)
                probs = self.dyn_w_proj.post_proj(probs, dws=post_proj_dw_args, query_vec=x, key_vec=x,
                                                  fast_infer=True)  # BN1S
                y = probs @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return y

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def _atten_context(query, key, value, atten_mask, pre_proj_dw_args, post_proj_dw_args):
    logits = query @ key.transpose(-2, -1)
    if pre_proj_dw_args is not None: logits = _cross_head_proj(logits, *pre_proj_dw_args)
    logits = torch.where(atten_mask, logits, torch.finfo(torch.float32).min)
    probs = logits.softmax(-1)
    if post_proj_dw_args is not None: probs = _cross_head_proj(probs, *post_proj_dw_args)
    o = probs @ value  # BNTS,BNSD->BNTD
    return o

def _cross_head_proj(inputs, sw, qw1, qw2, kw1, kw2, qdd, kdd, loop_over_dynamic_hd=False):
    out = inputs + torch.einsum('BNTS,NM->BMTS', inputs, sw) if sw is not None else inputs
    for i in range(2): # qw1.shape[-2]):
        qhidden = (inputs * qw1[..., i, :].transpose(-2, -1).unsqueeze(-1)).sum(1)  # BNTS,(BTN->BNT->BNT1)->BNTS->BTS
        qout = qhidden.unsqueeze(1) * qw2[..., i, :].transpose(-2, -1).unsqueeze(-1) # (BTS->B1TS),(BTN->BNT->BNT1)->BNTS
        out = out + qout
        khidden = (inputs * kw1[..., i, :].transpose(-2, -1).unsqueeze(-2)).sum(1)  # BNTS,(BSN->BNS->BN1S)->BNTS->BTS
        kout = khidden.unsqueeze(1) * kw2[..., i, :].transpose(-2, -1).unsqueeze(-2) # (BTS->B1TS),(BSN->BNS->BNS1)->BNTS
        out = out + kout
    qdout = inputs * qdd.transpose(-2, -1).unsqueeze(-1); out = out + qdout  # BNTS,(BTN->BNT->BNT1)->BNTS
    kdout = inputs * kdd.transpose(-2, -1).unsqueeze(-2); out = out + kdout  # BNTS,(BSN->BNS->BN1S)->BNTS
    return out

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

def make_window_mask(t, window_size):
    col_idx = torch.tile(torch.arange(t).unsqueeze(0), [t, 1])
    row_idx = torch.tile(torch.arange(t).unsqueeze(1), [1, t])
    bias_mask = (col_idx + window_size >= row_idx).tril().view(t, t)
    return bias_mask

def slice_dw(sw, qw1, qw2, kw1, kw2, qdd, kdd, start, stop, kv_start):
    return (sw,
            qw1[:, start : stop] if qw1 is not None else None,
            qw2[:, start : stop] if qw2 is not None else None,
            kw1[:, kv_start : stop] if kw1 is not None else None,
            kw2[:, kv_start : stop] if kw2 is not None else None,
            qdd[:, start : stop] if qdd is not None else None,
            kdd[:, kv_start : stop] if kdd is not None else None)

def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float32)

def unbind(ary, n, dim=0):
    return [torch.squeeze(a, dim=dim) for a in torch.split(ary, ary.shape[dim] // n, dim=dim)]

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, mode='half') -> Tensor:
    if mode == 'half':
        xshaped = x.float().reshape(*x.shape[:-1], 2,-1).transpose(-1,-2)
    elif mode == 'alternative':
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

class DynamicContextMultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, config, dropout=0.1):
        super(DynamicContextMultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.dyn_w_proj = DynamicWeightProjection(num_heads=h, query_input_dim=d_model, dynamic_squeeze_ratio=h // 2,  dynamic_w_hidden_dim=h * 4)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # Apply dynamic weight projection
        dynamic_weights = self.dyn_w_proj(query)

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout, dynamic_weights=dynamic_weights)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None, dynamic_weights=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply dynamic weights
        if dynamic_weights is not None:
            scores = scores * dynamic_weights

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
