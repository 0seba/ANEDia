from json import decoder
import os
import shutil
import asyncio
import numpy as np

from typing import Optional, Tuple
import torch
import torch.nn as nn

from ane_utils import print_compute_plan

from dia import config
from dia.layers import (
    DecoderLayer,
    Decoder,
)
from dia.config import DiaConfig, ModelConfig, EncoderConfig, DecoderConfig, DataConfig
from dia.ane_ops import (
    ANERMSNorm,
    update_kv_cache,
)
from dia.ane_layers import (
    ANERotaryEmbedding,
    ANEAttention,
    ANEMlpBlock,
    ANEDenseGeneral,
    apply_rotary_embedding,
)
from dia.state import create_attn_mask


class ANEDecoderLayer(nn.Module):
    def __init__(self, layer: "DecoderLayer"):
        super().__init__()
        self.layer = layer

        self.pre_sa_norm = ANERMSNorm(
            self.layer.pre_sa_norm,
            dim=1,
            w_num_unsqueezes=2,
        )
        self.pre_ca_norm = ANERMSNorm(
            self.layer.pre_ca_norm,
            dim=1,
            w_num_unsqueezes=2,
        )
        self.pre_mlp_norm = ANERMSNorm(
            self.layer.pre_mlp_norm,
            dim=1,
            w_num_unsqueezes=2,
        )

        self.self_attention = ANEAttention(self.layer.self_attention)
        self.cross_attention = ANEAttention(self.layer.cross_attention)
        self.mlp = ANEMlpBlock(self.layer.mlp)
        self.compute_dtype = self.layer.compute_dtype

    def forward(
        self,
        x: torch.Tensor,  # [B, D, 1, T] in NCHW format
        sin_q: torch.Tensor,  # [B, 1, D/2, T]
        cos_q: torch.Tensor,  # [B, 1, D/2, T]
        kv_write_index: torch.Tensor,
        kv_layer_write_idx: int,
        sin_k: Optional[torch.Tensor] = None,  # [B, 1, D/2, T]
        cos_k: Optional[torch.Tensor] = None,  # [B, 1, D/2, T]
        self_attn_mask: Optional[torch.Tensor] = None,  # [B, 1, T, S] or None
        cross_attn_mask: Optional[torch.Tensor] = None,  # [B, 1, T, S] or None
        enc_out: Optional[torch.Tensor] = None,  # [B, D, 1, S] in NCHW format
        self_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        slice_update_end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        # x = self.pre_sa_norm(x)
        if sin_k is None or cos_k is None:
            sin_k = sin_q
            cos_k = cos_q
        # Self-attention with KV cache
        x = self.self_attention(
            Xq=x,
            Xkv=x,
            sin_q=sin_q,
            cos_q=cos_q,
            sin_k=sin_k,
            cos_k=cos_k,
            attn_mask=self_attn_mask,
            cache=self_attn_cache,
            kv_write_idx=kv_write_index,
            kv_layer_write_idx=kv_layer_write_idx,
            slice_update_end=slice_update_end,
        )
        return x

        x = residual + x
        residual = x
        # x_norm = self.pre_ca_norm(x)
        # ca_out = self.cross_attention(
        #     Xq=x_norm,
        #     Xkv=enc_out,
        #     sin_q=sin_q,
        #     cos_q=cos_q,
        #     sin_k=sin_k,
        #     cos_k=cos_k,
        #     attn_mask=cross_attn_mask,
        #     cache=cross_attn_cache,
        #     kv_write_idx=0,
        #     kv_layer_write_idx=kv_layer_write_idx,
        # )
        # x = residual + ca_out

        residual = x
        # x = self.pre_mlp_norm(x)
        # x = self.mlp(x)
        x = residual + x

        return x


def create_cross_attn_mask_alt(
    q_padding_mask_1d: torch.Tensor,
    k_padding_mask_1d: torch.Tensor,
    device: torch.device,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Creates the attention mask (self or cross) mimicking JAX segment ID logic.
    """
    # B1, Tq = q_padding_mask_1d.shape
    # B2, Tk = k_padding_mask_1d.shape

    p_mask_q = q_padding_mask_1d.unsqueeze(2)  # Shape [B, Tq, 1]
    p_mask_k = k_padding_mask_1d.unsqueeze(1)  # Shape [B, 1, Tk]

    # Condition A: Non-padding query attends to non-padding key
    non_pad_attends_non_pad = p_mask_q * p_mask_k  # Shape [B, Tq, Tk]

    # Condition B: Padding query attends to padding key
    pad_attends_pad = (~p_mask_q) & (~p_mask_k)  # Shape [B, Tq, Tk]

    # Combine: True if padding status is compatible (both non-pad OR both pad)
    mask = torch.logical_or(
        non_pad_attends_non_pad, pad_attends_pad
    )  # Shape [B, Tq, Tk]

    if is_causal:
        # assert Tq == Tk, "Causal mask requires query and key sequence lengths to be equal"
        causal_mask_2d = torch.tril(
            torch.ones_like(mask[0], dtype=torch.bool, device=device)
        )  # Shape [B, Tq, Tk]
        causal_mask = mask & causal_mask_2d  # Shape [B, Tq, Tk]
        return causal_mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk]
    else:
        return mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk]


class ANEDecoder(nn.Module):
    def __init__(
        self, decoder: Decoder, batch_size: int = 2, audio_length: int | None = None
    ):
        super().__init__()
        self.decoder = decoder
        self.compute_dtype = decoder.layers[0].compute_dtype
        if audio_length is None:
            self.max_seq_len = decoder.config.data.audio_length
        else:
            self.max_seq_len = audio_length

        self.embeddings = decoder.embeddings
        self.rotary_emb = ANERotaryEmbedding(
            decoder.layers[0].self_attention.rotary_emb, max_seq_len=self.max_seq_len
        )
        self.layers = nn.ModuleList(
            [ANEDecoderLayer(layer) for layer in decoder.layers]
        )
        self.norm = ANERMSNorm(decoder.norm, dim=1, w_num_unsqueezes=2)
        self.logits_dense = ANEDenseGeneral(decoder.logits_dense)

        first_dim_size = min(batch_size * len(self.layers), 6)
        # first_dim_size = batch_size * len(self.layers)
        self_attention_kcache = torch.zeros(
            first_dim_size,
            self.decoder.config.model.decoder.kv_heads,
            self.max_seq_len,
            self.decoder.config.model.decoder.gqa_head_dim,
            dtype=self.compute_dtype,
        )
        self_attention_vcache = torch.zeros(
            first_dim_size,
            self.decoder.config.model.decoder.kv_heads,
            self.decoder.config.model.decoder.gqa_head_dim,
            self.max_seq_len,
            dtype=self.compute_dtype,
        )
        cross_attention_kcache = torch.zeros(
            first_dim_size,
            self.decoder.config.model.decoder.cross_query_heads,
            self.max_seq_len,
            self.decoder.config.model.decoder.cross_head_dim,
            dtype=self.compute_dtype,
        )
        cross_attention_vcache = torch.zeros(
            first_dim_size,
            self.decoder.config.model.decoder.cross_query_heads,
            self.decoder.config.model.decoder.cross_head_dim,
            self.max_seq_len,
            dtype=self.compute_dtype,
        )
        self.register_buffer("self_attention_kcache", self_attention_kcache)
        self.register_buffer("self_attention_vcache", self_attention_vcache)
        self.register_buffer("cross_attention_kcache", cross_attention_kcache)
        self.register_buffer("cross_attention_vcache", cross_attention_vcache)

    def forward(
        self,
        # x_ids: torch.Tensor,  # [B, T, C]
        x: torch.Tensor,  # [B, D, 1, T]  #  use embeddings as inputs instead of ids, move embeddings to separate model
        positions: torch.Tensor,  # [B, T]
        # sin_q: torch.Tensor,  # [B, D, 1, T]
        # cos_q: torch.Tensor,  # [B, D, 1, T]
        # kv_write_index: torch.Tensor,  # [1]
        # kv_write_end: torch.Tensor,  # [1]
        # encoder_lengths: torch.Tensor,  # [B, T]
        # enc_x_ids: torch.Tensor,  # [B, T]
        self_attn_mask: torch.Tensor | None = None,  # [B, 1, T, K]
        cross_attn_mask: torch.Tensor | None = None,  # [B, 1, T, S]
    ) -> torch.Tensor:
        # kv_write_index = torch.zeros(1, dtype=torch.int32)
        # slice_update_end = torch.full_like(kv_write_index, torch.tensor(1, dtype=torch.int32))
        # kv_write_index = kv_write_index[:, 0]
        kv_write_index = 0
        # slice_update_end = kv_write_end[:, 0]
        slice_update_end = x.size(-1) # + kv_write_index
        # encoder_lengths = encoder_lengths[:, 0]
    

        # make cross attention padding mask based on encoder_lengths
        # q_padding_mask_1d = torch.full_like(positions, 1, dtype=torch.bool)
        # k_padding_mask_1d = (
        #     torch.arange(self.max_seq_len, dtype=torch.int32) < encoder_lengths[:, 0]
        # )
        # cross_attn_mask = create_attn_mask(
        #     q_padding_mask_1d=q_padding_mask_1d,
        #     k_padding_mask_1d=k_padding_mask_1d,
        #     # k_padding_mask_1d=k_padding_mask_1d,
        #     device=x_ids.device,
        #     is_causal=False,
        # )
        # cross_attn_mask = cross_attn_mask.transpose(-1, -2)
        # cross_attn_mask = torch.where(
        #     cross_attn_mask,
        #     torch.tensor(0.0, dtype=self.compute_dtype),
        #     torch.tensor(-float("inf"), dtype=self.compute_dtype),
        # )

        # # make causal attention mask for safe attention
        # # for this dia tts it is alright to create it inside of the model
        # # but if you want to use speculative decoding
        # # it needs to be an input
        # self_attn_mask = (
        #     torch.arange(self.self_attention_kcache.shape[2], dtype=torch.int32)[
        #         None, None, None, :
        #     ]
        #     <= positions[:, None, :, None]
        # )
        # slice_update_end = x.size(-1) + kv_write_index
        # self_attn_mask = self_attn_mask.transpose(-1, -2)
        # self_attn_mask = torch.where(
        #     self_attn_mask,
        #     torch.tensor(0.0, dtype=self.compute_dtype),
        #     torch.tensor(-float("inf"), dtype=self.compute_dtype),
        # )

        sin_q, cos_q = self.rotary_emb(positions, permute_for_ane=True)
        sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)

        # x = None
        # x_ids = x_ids.split(1, -1)
        # for i in range(self.decoder.num_channels):
        #     channel_tokens = x_ids[i]
        #     channel_embed = self.embeddings[i](channel_tokens)
        #     x = channel_embed if x is None else x + channel_embed
        # x = x.permute(0, 3, 2, 1)  # [B, D, 1, T]
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_mask=self_attn_mask,
                # cross_attn_mask=cross_attn_mask,
                sin_q=sin_q,
                cos_q=cos_q,
                self_attn_cache=(
                    self.self_attention_kcache,
                    self.self_attention_vcache,
                ),
                cross_attn_cache=(
                    self.cross_attention_kcache,
                    self.cross_attention_vcache,
                ),
                kv_write_index=kv_write_index,
                kv_layer_write_idx=i,
                # slice_update_end=slice_update_end,
            )
        # x = self.norm(x)
        return x
        logits = self.logits_dense(x)  # move prediction head into separate model
        return logits

    def convert(
        self,
        save_path: str,
        seqlen: int = 32,
        batch_size: int = 2,
        remove_old: bool = False,
    ):
        import coremltools as ct
        import coremltools.converters.mil as mil

        input_lengths = [1, 32, 64, 128, 256, 384, 512, 768, 1024]
        input_def = mil.input_types.EnumeratedShapes(
            # shapes=[(batch_size, s, self.decoder.num_channels) for s in input_lengths]
            shapes=[
                (batch_size, self.decoder.config.model.decoder.n_embd, 1, s)
                for s in input_lengths
            ]
        )
        positions_shape_def = mil.input_types.EnumeratedShapes(
            shapes=[(batch_size, s) for s in input_lengths]
        )

        # length = 32
        batch = torch.ones(
            # (batch_size, seqlen, self.decoder.num_channels),
            # dtype=torch.int32,
            (batch_size, self.decoder.config.model.decoder.n_embd, 1, seqlen),
            dtype=torch.float16,
        )
        positions = torch.arange(seqlen, dtype=torch.int32)[None].expand(batch_size, seqlen)
        sin_q = torch.ones(
            (
                batch_size,
                1,
                self.decoder.config.model.decoder.gqa_head_dim // 2,
                seqlen,
            ),
            dtype=torch.float16,
        )
        cos_q = torch.ones(
            (
                batch_size,
                1,
                self.decoder.config.model.decoder.gqa_head_dim // 2,
                seqlen,
            ),
            dtype=torch.float16,
        )
        kv_write_index = torch.zeros((1, seqlen), dtype=torch.int32)
        slice_update_end = torch.zeros((1, seqlen), dtype=torch.int32) + seqlen
        # kv_write_index = torch.zeros((1,), dtype=torch.int32)
        encoder_lengths = torch.full((batch_size, seqlen), seqlen, dtype=torch.int32)
        self_attn_mask = torch.zeros(
            (batch_size, 1, seqlen, seqlen), dtype=torch.float16
        )
        # cross_attn_mask = torch.zeros(
        #     (batch_size, 1, seqlen, seqlen), dtype=torch.float16
        # )

        self.eval()
        with torch.inference_mode():
            traced_model = torch.jit.trace(
                self,
                (
                    batch,
                    positions,
                    # kv_write_index,
                    # slice_update_end,
                    # encoder_lengths,
                    # self_attn_mask,
                    # cross_attn_mask,
                ),
            )
        print(traced_model.graph)

        inputs = [
            ct.TensorType(
                # name="x_ids",
                # shape=input_def,
                # dtype=np.int32,
                name="x",
                shape=input_def,
                # shape=(batch_size, self.decoder.config.model.decoder.n_embd, 1, seqlen),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="positions",
                shape=positions_shape_def,
                dtype=np.int32,
            ),
            # ct.TensorType(
            #     name="sin_q",
            #     shape=(
            #         batch_size,
            #         1,
            #         self.decoder.config.model.decoder.gqa_head_dim // 2,
            #         seqlen,
            #     ),
            #     dtype=np.float16,
            # ),
            # ct.TensorType(
            #     name="cos_q",
            #     shape=(
            #         batch_size,
            #         1,
            #         self.decoder.config.model.decoder.gqa_head_dim // 2,
            #         seqlen,
            #     ),
            #     dtype=np.float16,
            # ),
            # ct.TensorType(
            #     name="kv_write_index",
            #     shape=mil.input_types.EnumeratedShapes(shapes=[(1, i) for i in input_lengths]),
            #     # shape=(1,),
            #     dtype=np.int32,
            # ),
            # ct.TensorType(
            #     name="slice_update_end",
            #     shape=mil.input_types.EnumeratedShapes(shapes=[(1, i) for i in input_lengths]),
            #     # shape=(batch_size,),
            #     dtype=np.int32,
            # ),
            # ct.TensorType(
            #     name="encoder_lengths",
            #     shape=mil.input_types.EnumeratedShapes(shapes=[(batch_size, i) for i in input_lengths]),
            #     # shape=(batch_size,),
            #     dtype=np.int32,
            # ),
            # ct.TensorType(
            #     name="self_attn_mask",
            #     # shape=mil.input_types.EnumeratedShapes(shapes=[(batch_size, i) for i in input_lengths]),
            #     shape=(batch_size, 1, seqlen, self.self_attention_kcache.shape[2]),
            #     dtype=np.float16,
            # ),
            # ct.TensorType(
            #     name="cross_attn_mask",
            #     # shape=mil.input_types.EnumeratedShapes(shapes=[(batch_size, i) for i in input_lengths]),
            #     shape=(batch_size, 1, seqlen, self.cross_attention_kcache.shape[2]),
            #     dtype=np.float16,
            # ),
        ]

        outputs = [
            ct.TensorType(
                name="logits",
                dtype=np.float16,
            ),
        ]

        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.self_attention_kcache.size()),
                name="self_attention_kcache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.self_attention_vcache.size()),
                name="self_attention_vcache",
            ),
            # ct.StateType(
            #     wrapped_type=ct.TensorType(shape=self.cross_attention_kcache.size()),
            #     name="cross_attention_kcache",
            # ),
            # ct.StateType(
            #     wrapped_type=ct.TensorType(shape=self.cross_attention_vcache.size()),
            #     name="cross_attention_vcache",
            # ),
        ]

        mlmodel: ct.models.MLModel = ct.convert(
            traced_model,
            convert_to="milinternal",
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
            # pass_pipeline=pipeline,
        )
        print(mlmodel)
        # mlmodel.export_as_multifunction = True
        mlmodel: ct.models.MLModel = ct.convert(
            mlmodel,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=False,
        )
        # delete directory if it exists
        mlmodelc_path = save_path.rstrip(".mlpackage") + ".mlmodelc"
        if remove_old:
            if os.path.isdir(save_path.rstrip(".mlpackage") + ".mlpackage"):
                shutil.rmtree(save_path.rstrip(".mlpackage") + ".mlpackage")
            if os.path.isdir(mlmodelc_path):
                shutil.rmtree(mlmodelc_path)

        mlmodel.save(save_path)
        # copy compiled model
        compiled_path = mlmodel.get_compiled_model_path()
        shutil.copytree(
            compiled_path,
            mlmodelc_path,
            dirs_exist_ok=True,
        )

        print_compute_plan(mlmodelc_path)

        # print(
        #     mlmodel.predict(
        #         {
        #             # "x_ids": np.ones((batch_size, seqlen, self.decoder.num_channels), dtype=np.int32),
        #             "x": np.ones(
        #                 (
        #                     batch_size,
        #                     self.decoder.config.model.decoder.n_embd,
        #                     1,
        #                     seqlen,
        #                 ),
        #                 dtype=np.float16,
        #             ),
        #             "sin_q": np.ones(
        #                 (
        #                     batch_size,
        #                     1,
        #                     self.decoder.config.model.decoder.gqa_head_dim // 2,
        #                     seqlen,
        #                 ),
        #                 dtype=np.float16,
        #             ),
        #             "cos_q": np.ones(
        #                 (
        #                     batch_size,
        #                     1,
        #                     self.decoder.config.model.decoder.gqa_head_dim // 2,
        #                     seqlen,
        #                 ),
        #                 dtype=np.float16,
        #             ),
        #             "kv_write_index": np.zeros((1,), dtype=np.int32),
        #             # "encoder_lengths": np.full((batch_size,), seqlen, dtype=np.int32),
        #             "self_attn_mask": np.zeros(
        #                 (batch_size, 1, seqlen, self.self_attention_kcache.shape[2]),
        #                 dtype=np.float16,
        #             ),
        #             # "cross_attn_mask": np.zeros(
        #             #     (batch_size, 1, seqlen, self.cross_attention_kcache.shape[2]),
        #             #     dtype=np.float16,
        #             # ),
        #         },
        #         mlmodel.make_state(),
        #     )
        # )


class PreComputeCrossAttentionCache(nn.Module):
    def __init__(self, decoder: ANEDecoder, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.decoder = decoder
        self.config = decoder.decoder.config
        self.num_layers = len(decoder.layers)

        self_attention_kcache = torch.zeros(
            batch_size * self.num_layers,
            self.config.model.decoder.kv_heads,
            self.config.data.audio_length,
            self.config.model.decoder.gqa_head_dim,
        )
        self_attention_vcache = torch.zeros(
            batch_size * self.num_layers,
            self.config.model.decoder.kv_heads,
            self.config.model.decoder.gqa_head_dim,
            self.config.data.audio_length,
        )
        cross_attention_kcache = torch.zeros(
            batch_size * self.num_layers,
            self.config.model.decoder.cross_query_heads,
            self.config.data.text_length,
            self.config.model.decoder.cross_head_dim,
        )
        cross_attention_vcache = torch.zeros(
            batch_size * self.num_layers,
            self.config.model.decoder.cross_query_heads,
            self.config.model.decoder.cross_head_dim,
            self.config.data.text_length,
        )
        self.register_buffer("self_attn_key_cache", self_attention_kcache)
        self.register_buffer("self_attn_value_cache", self_attention_vcache)
        self.register_buffer("cross_attn_key_cache", cross_attention_kcache)
        self.register_buffer("cross_attn_value_cache", cross_attention_vcache)

    def forward(
        self,
        enc_out: torch.Tensor,  # (B, E, 1, S)
        enc_positions: torch.Tensor | None = None,  # (B, S)
    ) -> torch.Tensor:
        if enc_positions is None:
            shape = enc_out[:, 0, 0]
            enc_positions = (
                torch.cumsum(torch.ones_like(shape), dim=1, dtype=torch.int32) - 1
            )  # to suppor enumerated shapes

        sin_q, cos_q = self.decoder.rotary_emb(enc_positions, permute_for_ane=True)
        sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)

        for i, layer in enumerate(self.decoder.layers):
            cross_attn_module = layer.cross_attention
            k_proj = cross_attn_module.k_proj(enc_out)
            v_proj = cross_attn_module.v_proj(enc_out)

            k_proj = k_proj.view(
                k_proj.size(0),
                self.config.model.decoder.cross_query_heads,
                self.config.model.decoder.cross_head_dim,
                k_proj.size(-1),
            )
            k_rotated = apply_rotary_embedding(k_proj, sin_q, cos_q)
            k_rotated = k_rotated.permute(0, 1, 3, 2)
            v_proj = v_proj.view(
                v_proj.size(0),
                self.config.model.decoder.cross_query_heads,
                self.config.model.decoder.cross_head_dim,
                v_proj.size(-1),
            )
            v_rotated = apply_rotary_embedding(v_proj, sin_q, cos_q)

            update_kv_cache(
                k_rotated,
                v_rotated,
                (self.cross_attn_key_cache, self.cross_attn_value_cache),
                0,
                self.batch_size * i,
            )

        ##### To make the state ops run on ANE we need to read after writing
        readk = self.cross_attn_key_cache[
            self.batch_size * i : self.batch_size * i + 1
        ]
        readk = readk.split(1, 1)[0]
        readv = self.cross_attn_value_cache[
            self.batch_size * i : self.batch_size * i + 1
        ]
        readv = readv.split(1, 1)[0]
        # now read self attention even though we did nothing for coreml support
        readk2 = self.self_attn_key_cache[
            self.batch_size * i : self.batch_size * i + 1
        ]
        readk2 = readk2.split(1, 1)[0]
        readv2 = self.self_attn_value_cache[
            self.batch_size * i : self.batch_size * i + 1
        ]
        readv2 = readv2.split(1, 1)[0]

        return readk, readv, readk2, readv2

    def convert(self, save_path: str):
        import coremltools as ct
        import coremltools.converters.mil as mil

        hidden_dim = self.config.model.encoder.n_embd

        # enumerated shapes for input
        lengths = [32, 64, 128, 256, 384, 512, 768, 1024]
        input_shapes = mil.input_types.EnumeratedShapes(
            shapes=[(self.batch_size, hidden_dim, 1, s) for s in lengths]
        )
        inputs = [
            ct.TensorType(
                name="enc_out",
                shape=input_shapes,
                dtype=np.float16,
            ),
        ]
        outputs = [
            ct.TensorType(
                name="sumk",
                dtype=np.float16,
            ),
            ct.TensorType(
                name="sumv",
                dtype=np.float16,
            ),
            ct.TensorType(
                name="sumk2",
                dtype=np.float16,
            ),
            ct.TensorType(
                name="sumv2",
                dtype=np.float16,
            ),
        ]
        # outputs = [ct.TensorType(name=f"kreads_{i}", dtype=np.float16) for i in range(len(self.decoder.layers))]
        # outputs += [ct.TensorType(name=f"vreads_{i}", dtype=np.float16) for i in range(len(self.decoder.layers))]
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.cross_attn_key_cache.size()),
                name="cross_attn_key_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.cross_attn_value_cache.size()),
                name="cross_attn_value_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.self_attn_key_cache.size()),
                name="self_attn_key_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.self_attn_value_cache.size()),
                name="self_attn_value_cache",
            ),
        ]

        with torch.inference_mode():
            example_kwarg_inputs = {
                "enc_out": torch.zeros(
                    (self.batch_size, hidden_dim, 1, 32), dtype=torch.float16
                ),
                # "enc_positions": torch.zeros((2, 32), dtype=torch.int32),
            }
            traced_model = torch.jit.trace(
                self, example_kwarg_inputs=example_kwarg_inputs
            )

        mlmodel = ct.convert(
            traced_model,
            convert_to="milinternal",
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
        )
        print(mlmodel)
        # mlmodel.export_as_multifunction = True
        mlmodel: ct.models.MLModel = ct.convert(
            mlmodel,
            # convert_to="milinternal",
            inputs=inputs,
            outputs=outputs,
            # states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            # compute_precision=ct.precision.FLOAT16,
            skip_model_load=False,
            # pass_pipeline=pipeline,
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.save(save_path)
        # copy compiled model
        compiled_path = mlmodel.get_compiled_model_path()
        shutil.copytree(
            compiled_path,
            save_path.rstrip(".mlpackage") + ".mlmodelc",
            dirs_exist_ok=True,
        )
        print_compute_plan(compiled_path)


class ReadState(nn.Module):
    def __init__(self, decoder: ANEDecoder, batch_size: int = 2):
        super().__init__()
        self.decoder = decoder
        self.batch_size = batch_size
        self.config = decoder.decoder.config
        self.num_layers = len(decoder.layers)

        kcache = torch.zeros(
            batch_size * self.num_layers,
            self.config.model.decoder.cross_query_heads,
            self.config.data.text_length,
            self.config.model.decoder.cross_head_dim,
        )
        vcache = torch.zeros(
            batch_size * self.num_layers,
            self.config.model.decoder.cross_query_heads,
            self.config.model.decoder.cross_head_dim,
            self.config.data.text_length,
        )
        self.register_buffer("kcache", kcache)
        self.register_buffer("vcache", vcache)

    def forward(self):
        ksize = self.kcache.size()
        vsize = self.vcache.size()
        readk = self.kcache[0 : ksize[0]]
        readv = self.vcache[0 : vsize[0]]
        return readk, readv

    def convert(self, save_path: str):
        import shutil
        import numpy as np
        import coremltools as ct
        import coremltools.converters.mil as mil

        hidden_dim = self.config.model.encoder.n_embd
        inputs = [
            # ct.TensorType(
            #     name="enc_out",
            #     shape=input_shapes,
            #     dtype=np.float16,
            # ),
        ]
        outputs = [
            ct.TensorType(
                name="read_kcache",
                dtype=np.float16,
            ),
            ct.TensorType(
                name="read_vcache",
                dtype=np.float16,
            ),
        ]
        # outputs = [ct.TensorType(name=f"kreads_{i}", dtype=np.float16) for i in range(len(self.decoder.layers))]
        # outputs += [ct.TensorType(name=f"vreads_{i}", dtype=np.float16) for i in range(len(self.decoder.layers))]
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.kcache.size()),
                name="kcache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.vcache.size()),
                name="vcache",
            ),
        ]

        with torch.inference_mode():
            traced_model = torch.jit.trace(self, example_kwarg_inputs={})

        print(traced_model.graph)

        mlmodel = ct.convert(
            traced_model,
            convert_to="milinternal",
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
        )
        print(mlmodel)
        # mlmodel.export_as_multifunction = True
        mlmodel: ct.models.MLModel = ct.convert(
            mlmodel,
            # convert_to="milinternal",
            inputs=inputs,
            outputs=outputs,
            # states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            # compute_precision=ct.precision.FLOAT16,
            skip_model_load=False,
            # pass_pipeline=pipeline,
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.save(save_path)
        # copy compiled model
        compiled_path = mlmodel.get_compiled_model_path()
        shutil.copytree(
            compiled_path,
            save_path.rstrip(".mlpackage") + ".mlmodelc",
            dirs_exist_ok=True,
        )


if __name__ == "__main__":
    from dia.model import DiaModel
    import shutil
    import numpy as np
    import coremltools as ct
    import coremltools.converters.mil as mil

    # load pretrained dia model
    repo_id = "seba/Dia-1.6B-float16"
    model = DiaModel.from_pretrained(repo_id, compute_dtype=torch.float16)
    batch_size = 1
    ane_decoder = ANEDecoder(model.decoder, batch_size=batch_size, audio_length=512)
    # ane_decoder.layers = ane_decoder.layers[:1]
    # ane_decoder.convert("dia_decoder", 1, batch_size=batch_size, remove_old=True)
    precompute_cross_attn_cache = PreComputeCrossAttentionCache(ane_decoder)
    precompute_cross_attn_cache.eval()
    precompute_cross_attn_cache.convert("precompute_cross_attn_cache")

    # read_state = ReadState(ane_decoder)
    # read_state.eval()
    # read_state.convert("read_state")
