from json import decoder
import os
import time
import shutil
import asyncio
import numpy as np

from typing import Optional, Tuple, List
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
    simple_attention,
)
from dia.ane_layers import (
    ANERotaryEmbedding,
    ANEAttention,
    ANEMlpBlock,
    ANEDenseGeneral,
    apply_rotary_embedding,
)
from dia.state import create_attn_mask


class ANEDecoderSelfAttention(ANEAttention):
    def forward(
        self,
        Xq: torch.Tensor,
        Xkv: torch.Tensor,
        sin_q: torch.Tensor,
        cos_q: torch.Tensor,
        sin_k: torch.Tensor,
        cos_k: torch.Tensor,
        kv_write_idx: torch.Tensor,
        kv_layer_idx: int,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size = Xq.shape[0]
        seqlen_q = Xq.shape[-1]

        # (B, D, 1, S) -> (B, N*H, 1, S) for query
        q_proj = self.q_proj(Xq)
        q_proj = q_proj.view(
            batch_size,
            self.num_query_heads,
            self.head_dim,
            seqlen_q,
        )  # (B, N, H, S)
        q_rotated = apply_rotary_embedding(q_proj, sin_q, cos_q)

        seqlen_k = Xkv.shape[-1]
        k_proj = self.k_proj(Xkv)  # (B, H*D, 1, S)
        k_proj = k_proj.view(
            batch_size, self.num_kv_heads, self.head_dim, seqlen_k
        )  # (B, H, D, S)
        k_rotated = apply_rotary_embedding(k_proj, sin_k, cos_k)
        k_rotated = k_rotated.permute(0, 1, 3, 2)  # (B, H, S, D)
        v_proj = self.v_proj(Xkv).view(
            batch_size, self.num_kv_heads, self.head_dim, seqlen_k
        )  # (B, H, D, S)
        if cache is not None:
            update_kv_cache(
                k_rotated,
                v_proj,
                cache,
                kv_write_idx,
                kv_layer_idx,
            )
            k_cache, v_cache = cache
            # k_cache = k_cache.split(batch_size, dim=0)
            # v_cache = v_cache.split(batch_size, dim=0)
            # key = k_cache[kv_layer_idx // batch_size]
            # value = v_cache[kv_layer_idx // batch_size]

            key = k_cache[kv_layer_idx : kv_layer_idx + batch_size]
            value = v_cache[kv_layer_idx : kv_layer_idx + batch_size]
        else:
            key = k_rotated
            value = v_proj

        attention = simple_attention(
            q_rotated,
            key,
            value,
            attn_mask,
        )
        if isinstance(attention, list):
            attention = torch.cat(attention, dim=1)

        return self.o_proj(attention)


class ANEDecoderLayer(nn.Module):
    def __init__(self, layer: "DecoderLayer"):
        super().__init__()
        self.layer = layer
        self.self_attention = ANEDecoderSelfAttention(layer.self_attention)
        self.cross_attention = ANEAttention(layer.cross_attention)
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
        self.mlp = ANEMlpBlock(self.layer.mlp)

    def forward(
        self,
        x: torch.Tensor,
        sin_q: torch.Tensor,
        cos_q: torch.Tensor,
        sin_k: torch.Tensor,
        cos_k: torch.Tensor,
        kv_write_idx: torch.Tensor,
        kv_layer_idx: int,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        self_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        residual = x
        x = self.pre_sa_norm(x)
        x = self.self_attention(
            Xq=x,
            Xkv=x,
            sin_q=sin_q,
            cos_q=cos_q,
            sin_k=sin_k,
            cos_k=cos_k,
            kv_write_idx=kv_write_idx,
            kv_layer_idx=kv_layer_idx,
            attn_mask=self_attn_mask,
            cache=self_attn_cache,
        )
        x = residual + x
        residual = x
        x = self.pre_ca_norm(x)
        x = self.cross_attention(
            x,
            x,
            sin_q=sin_q,
            cos_q=cos_q,
            sin_k=sin_k,
            cos_k=cos_k,
            kv_write_idx=kv_write_idx,
            kv_layer_idx=kv_layer_idx,
            attn_mask=cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + x
        residual = x
        x = self.pre_mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class ANEDecoder(nn.Module):
    def __init__(
        self,
        decoder: "Decoder",
        batch_size: int = 2,
        audio_length: int | None = None,
        layer_from: int = 0,
        layer_to: int = -1,
    ):
        super().__init__()
        self.layer_from = layer_from
        self.layer_to = layer_to if layer_to > 0 else len(decoder.layers)
        self.apply_final_norm = layer_to == len(decoder.layers)
        self.decoder = decoder
        self.batch_size = batch_size
        self.config = decoder.config
        self.decoder_config = decoder.config.model.decoder
        self.audio_length = (
            audio_length
            if audio_length is not None
            else decoder.config.data.audio_length
        )

        self.norm = ANERMSNorm(
            self.decoder.norm,
            dim=1,
            w_num_unsqueezes=2,
        )
        self.layers = nn.ModuleList()
        # for layer in self.decoder.layers[self.layer_from : self.layer_to]:
        for layer in self.decoder.layers:
            self.layers.append(ANEDecoderLayer(layer))
        self.rotary_emb = ANERotaryEmbedding(
            self.decoder.layers[0].self_attention.rotary_emb,
            max_seq_len=self.audio_length,
        )
        self.compute_dtype = self.decoder.layers[0].compute_dtype

        self_attn_key_cache = torch.zeros(
            batch_size * len(self.layers),
            self.decoder_config.kv_heads,
            self.audio_length,
            self.decoder_config.gqa_head_dim,
            dtype=torch.float16,
        )
        self_attn_value_cache = torch.zeros(
            batch_size * len(self.layers),
            self.decoder_config.kv_heads,
            self.decoder_config.gqa_head_dim,
            self.audio_length,
            dtype=torch.float16,
        )
        cross_attn_key_cache = torch.zeros(
            batch_size * len(self.layers),
            self.config.model.encoder.n_head,
            self.config.data.text_length,
            self.config.model.encoder.head_dim,
            dtype=torch.float16,
        )
        cross_attn_value_cache = torch.zeros(
            batch_size * len(self.layers),
            self.config.model.encoder.n_head,
            self.config.model.encoder.head_dim,
            self.config.data.text_length,
            dtype=torch.float16,
        )
        self.register_buffer("self_attn_key_cache", self_attn_key_cache)
        self.register_buffer("self_attn_value_cache", self_attn_value_cache)
        self.register_buffer("cross_attn_key_cache", cross_attn_key_cache)
        self.register_buffer("cross_attn_value_cache", cross_attn_value_cache)

    def forward(
        self,
        x: torch.Tensor,  # (B, D, 1, T)
        positions: torch.Tensor,  # (B, T)
        kv_write_index: torch.Tensor,  # (B, T)
        self_attn_mask: torch.Tensor,  # (B, 1, T, C)
        encoder_lengths: torch.Tensor,  # (B)
    ) -> torch.Tensor:
        # if kv_write_index.size(0) > 1:
        # kv_write_index = kv_write_index[[0]]
        # kv_write_index = 0
        # kv_write_index = torch.sum(kv_write_index)
        # if encoder_lengths.size(1) > 1:
        encoder_lengths = encoder_lengths[:, 0:1]
        # kv_write_index = kv_write_index[:, 0]
        # positions = torch.ones_like(x[:, 0].squeeze(1), dtype=torch.int32)
        # positions = torch.cumsum(positions, dim=-1) - 1
        sin_q, cos_q = self.rotary_emb(positions, permute_for_ane=True)
        sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)
        self_attn_mask = self_attn_mask.transpose(-1, -2)

        q_padding_mask_1d = (
            torch.full_like(positions, 1, dtype=torch.int32) == 1
        )  # had issues with simple torch.bool
        k_padding_mask_1d = (
            torch.arange(self.config.data.text_length, dtype=torch.int32).unsqueeze(0)
            < encoder_lengths
        )
        cross_attn_mask = create_attn_mask(
            q_padding_mask_1d=q_padding_mask_1d,
            k_padding_mask_1d=k_padding_mask_1d,
            device=x.device,
            is_causal=False,
        )
        cross_attn_mask = cross_attn_mask.transpose(-1, -2)
        cross_attn_mask = torch.where(
            cross_attn_mask,
            torch.tensor(0.0, dtype=self.compute_dtype),
            torch.tensor(-float("inf"), dtype=self.compute_dtype),
        )

        # for i in range(len(self.layers)):
        for i in range(self.layer_from, self.layer_to):
            x = self.layers[i](
                x=x,
                sin_q=sin_q,
                cos_q=cos_q,
                sin_k=sin_q,
                cos_k=cos_q,
                kv_write_idx=kv_write_index,
                # kv_layer_idx=(i + self.layer_from) * self.batch_size,
                kv_layer_idx=i * self.batch_size,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                self_attn_cache=(self.self_attn_key_cache, self.self_attn_value_cache),
                cross_attn_cache=(
                    self.cross_attn_key_cache,
                    self.cross_attn_value_cache,
                ),
            )

        if self.apply_final_norm:
            x = self.norm(x)
        return x

    def convert(
        self,
        seqlen: List[int],
        save_path: str = "ane_decoder.mlpackage",
        remove_old: bool = True,
        skip_model_load: bool = True,
    ):
        import coremltools as ct

        self.eval()
        tracing_seqlen = 2
        x = torch.randn(
            self.batch_size,
            self.decoder_config.n_embd,
            1,
            tracing_seqlen,
            dtype=torch.float16,
        )
        # kv_write_index = torch.ones(self.batch_size, tracing_seqlen, dtype=torch.int32)
        kv_write_index = torch.ones(1, dtype=torch.int32)
        positions = (
            torch.arange(tracing_seqlen, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(self.batch_size, 1)
        )
        self_attn_mask = torch.zeros(
            self.batch_size, 1, tracing_seqlen, self.audio_length, dtype=torch.float16
        )
        encoder_lengths = torch.ones(self.batch_size, 1, dtype=torch.int32)

        with torch.inference_mode():
            traced_model = torch.jit.trace(
                self, [x, positions, kv_write_index, self_attn_mask, encoder_lengths]
            )

        enum_shapes = seqlen
        kv_write_index_shape = (1,)
        encoder_lengths_shape = (self.batch_size, 1)
        if len(enum_shapes) == 1:
            x_shape = (self.batch_size, self.decoder_config.n_embd, 1, enum_shapes[0])
            positions_shape = (self.batch_size, enum_shapes[0])
            self_attn_mask_shape = (
                self.batch_size,
                1,
                enum_shapes[0],
                self.audio_length,
            )
        else:
            x_shape = ct.EnumeratedShapes(
                [
                    (self.batch_size, self.decoder_config.n_embd, 1, s)
                    for s in enum_shapes
                ]
            )
            positions_shape = ct.EnumeratedShapes(
                [(self.batch_size, s) for s in enum_shapes]
            )
            self_attn_mask_shape = ct.EnumeratedShapes(
                [
                    (
                        self.batch_size,
                        1,
                        s,
                        self.audio_length,
                    )
                    for s in enum_shapes
                ]
            )
            # kv_write_index_shape = ct.EnumeratedShapes([(s,) for s in enum_shapes])
            # encoder_lengths_shape = ct.EnumeratedShapes(
            #     [(self.batch_size, s) for s in enum_shapes]
            # )
        inputs = [
            ct.TensorType(
                name="x",
                shape=x_shape,
                dtype=np.float16,
            ),
            ct.TensorType(
                name="positions",
                shape=positions_shape,
                dtype=np.int32,
            ),
            ct.TensorType(
                name="kv_write_index",
                shape=kv_write_index_shape,
                dtype=np.int32,
            ),
            ct.TensorType(
                name="self_attn_mask",
                shape=self_attn_mask_shape,
                dtype=np.float16,
            ),
            ct.TensorType(
                name="encoder_lengths",
                shape=encoder_lengths_shape,
                dtype=np.int32,
            ),
        ]
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.self_attn_key_cache.size()),
                name="self_attn_key_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.self_attn_value_cache.size()),
                name="self_attn_value_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.cross_attn_key_cache.size()),
                name="cross_attn_key_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.cross_attn_value_cache.size()),
                name="cross_attn_value_cache",
            ),
        ]
        outputs = [
            ct.TensorType(
                name="hidden_states",
                dtype=np.float16,
            ),
            # ct.TensorType(
            #     name="readk",
            #     dtype=np.float16,
            # ),
            # ct.TensorType(
            #     name="readv",
            #     dtype=np.float16,
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
        )
        # mlmodel.export_as_multifunction = True
        print(mlmodel)
        mlmodel = ct.convert(
            mlmodel,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=skip_model_load,
        )

        # function_name_to_materialization_map = {
        #     # "function_name_to_materialization_map": {
        #     #     "materialization_2_3": {"input_ids": (1, 2), "mask": (2, 3)},
        #     # }
        # }
        # for length in enum_shapes:
        #     # function_name_to_materialization_map[
        #     #     "function_name_to_materialization_map"
        #     # ][f"decoder_{length}"] = {
        #     function_name_to_materialization_map[f"decoder_{length}"] = {
        #         "x": (self.batch_size, 2048, 1, length),
        #         "positions": (self.batch_size, length),
        #         "kv_write_index": (1,),
        #         "self_attn_mask": (self.batch_size, 1, length, self.audio_length),
        #         "encoder_lengths": (self.batch_size, 1),
        #     }
        # function_name_to_materialization_map["main"] = function_name_to_materialization_map["decoder_1"]  # overwrite flexible inputs to prevent any issue

        save_path = save_path.rstrip(".mlpackage") + ".mlpackage"
        mlmodelc_path = save_path.rstrip(".mlpackage") + ".mlmodelc"
        if remove_old:
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
            if os.path.isdir(mlmodelc_path):
                shutil.rmtree(mlmodelc_path)

        # ct.utils.materialize_dynamic_shape_mlmodel(
        #     mlmodel,
        #     function_name_to_materialization_map,
        #     save_path,
        # )
        mlmodel.save(save_path)
        # time model load time
        # start = time.time()
        # mlmodel = ct.models.MLModel(
        #     save_path, compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=False
        # )
        # end = time.time()
        # print("Model load time:", end - start)

        # start = time.time()
        # ct.utils.compile_model(mlmodel, mlmodelc_path)
        # end = time.time()
        # print("Model compile time:", end - start)

        # copy compiled model
        if not skip_model_load:
            compiled_path = mlmodel.get_compiled_model_path()
            # shutil.copytree(
            #     compiled_path,
            #     mlmodelc_path,
            #     dirs_exist_ok=True,
            # ) 

            print_compute_plan(compiled_path)
            # kv_write_index = np.ones((1,), dtype=np.int32)
            # if len(enum_shapes) > 1:
            # kv_write_index = np.ones((1, 1), dtype=np.int32)

            start = time.time()
            mlmodel = ct.models.CompiledMLModel(
                compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            end = time.time()
            # print("Model load time:", end - start)

            start = time.time()
            print(
                mlmodel.predict(
                    {
                        "x": np.ones(
                            (
                                self.batch_size,
                                self.decoder.config.model.decoder.n_embd,
                                1,
                                1,
                            ),
                            dtype=np.float16,
                        ),
                        "positions": np.full((self.batch_size, 1), 32, dtype=np.int32),
                        "kv_write_index": np.ones((1,), dtype=np.int32),
                        "self_attn_mask": np.zeros(
                            (self.batch_size, 1, 1, self.self_attn_key_cache.shape[2]),
                            dtype=np.float16,
                        ),
                        "encoder_lengths": np.full(
                            (self.batch_size, 1), 32, dtype=np.int32
                        ),
                    },
                    mlmodel.make_state(),
                )
            )
            end = time.time()
            print("Model predict time:", end - start)


if __name__ == "__main__":
    from dia.model import DiaModel
    import shutil
    import numpy as np

    # batch size as command line argument
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_file", type=str, default="dia_decoder")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--layer_from", type=int, default=0)
    parser.add_argument("--layer_to", type=int, default=-1)
    parser.add_argument("--audio_length", type=int, default=1024)
    parser.add_argument(
        "--seqlen",
        nargs="+",
        type=int,
        default=[1, 8, 32, 128],
    )
    args = parser.parse_args()
    batch_size = args.batch_size
    audio_length = args.audio_length

    # load pretrained dia model
    repo_id = "seba/Dia-1.6B-float16"
    model = DiaModel.from_pretrained(repo_id, compute_dtype=torch.float16)
    ane_decoder = ANEDecoder(
        model.decoder,
        batch_size=batch_size,
        audio_length=audio_length,
        layer_from=args.layer_from,
        layer_to=args.layer_to,
    )
    ane_decoder.convert(args.seqlen, args.save_file, remove_old=True, skip_model_load=not args.load_model)
