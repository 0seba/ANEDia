from json import encoder
from typing import Optional
import torch
import torch.nn as nn

from dia.layers import (
    EncoderLayer,
    Encoder,
)
from dia.config import DiaConfig
from dia.ane_ops import (
    ANERMSNorm,
)
from dia.ane_layers import (
    ANERotaryEmbedding,
    ANEAttention,
    ANEMlpBlock,
)
from dia.state import create_attn_mask


class ANEEncoderLayer(nn.Module):
    def __init__(self, layer: EncoderLayer):
        super().__init__()
        self.layer = layer
        self.pre_sa_norm = ANERMSNorm(
            self.layer.pre_sa_norm,
            dim=1,
            w_num_unsqueezes=2,
        )
        self.self_attention = ANEAttention(
            self.layer.self_attention,
        )
        self.post_sa_norm = ANERMSNorm(
            self.layer.post_sa_norm,
            dim=1,
            w_num_unsqueezes=2,
        )
        self.mlp = ANEMlpBlock(
            self.layer.mlp,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        sin_q: Optional[torch.Tensor] = None,
        cos_q: Optional[torch.Tensor] = None,
    ):
        residual = x
        x = self.pre_sa_norm(x)
        x = self.self_attention(
            x,
            x,
            attn_mask=attn_mask,
            sin_q=sin_q,
            cos_q=cos_q,
            sin_k=sin_q,
            cos_k=cos_q,
        )
        x = x + residual
        residual = x
        x = self.post_sa_norm(x)
        x = self.mlp(x)
        return residual + x


class ANEEncoder(nn.Module):
    """
    ANE-friendly implementation of the Encoder module.

    This implementation:
    1. Uses ANE-friendly operations and layouts
    2. Maintains NCHW format throughout
    3. Uses precomputed rotary embeddings
    """

    def __init__(self, encoder: Encoder, max_seq_len: int = 4096):
        super().__init__()
        self.encoder = encoder
        self.max_seq_len = max_seq_len

        # Create ANE-compatible layers
        self.embedding = encoder.embedding  # Keep original embedding for now
        self.layers = nn.ModuleList(
            [ANEEncoderLayer(layer) for layer in encoder.layers]
        )
        self.norm = ANERMSNorm(encoder.norm, dim=1, w_num_unsqueezes=2)

        # Pre-compute rotary embeddings
        self.rotary_emb = ANERotaryEmbedding(
            encoder.layers[0].self_attention.rotary_emb, max_seq_len=max_seq_len
        )

    def forward(
        self,
        x_ids: torch.Tensor,  # [B, T]
        positions: torch.Tensor = None,  # [B, T]
        padding_mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> torch.Tensor:
        """
        Args:
            x_ids: Input token IDs [B, T]
            padding_mask: Boolean mask where True indicates padding tokens [B, T]

        Returns:
            Tensor of shape [B, T, D] in channels-last format
        """
        if positions is None:
            positions = torch.ones_like(x_ids)
            positions = torch.cumsum(positions, dim=1) - 1 # theoretically it should be the same without substraction because rope is relative
        
        # Get rotary embeddings [B, 1, D/2, T]
        sin_q, cos_q = self.rotary_emb(positions, permute_for_ane=True)
        sin_q, cos_q = sin_q.unsqueeze(1), cos_q.unsqueeze(1)

        # Create attention mask [B, 1, T, T]
        if padding_mask is None:
            padding_mask = x_ids != self.encoder.config.data.text_pad_value
        attn_mask = create_attn_mask(
            q_padding_mask_1d=padding_mask,
            k_padding_mask_1d=padding_mask,
            device=x_ids.device,
            is_causal=False,
        )
        attn_mask = attn_mask.transpose(-1, -2)
        x = self.embedding(x_ids)  # [B, T, D]
        x = x.transpose(1, 2).unsqueeze(2)  # [B, D, 1, T]
        attn_mask = torch.where(
            attn_mask,
            torch.tensor(0.0, dtype=x.dtype),
            torch.tensor(-float("inf"), dtype=x.dtype),
        )
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                attn_mask=attn_mask,
                sin_q=sin_q,
                cos_q=cos_q,
            )
        x = self.norm(x)

        return x

def convert(config_path: str, tensors_path: str, save_path: str):
    from safetensors.torch import load_file
    
    config = DiaConfig.load(config_path)
    torch_encoder = Encoder(config, compute_dtype=torch.float16)
    state_dict = load_file(tensors_path)
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state_dict[key[len("encoder."):]] = value
    torch_encoder.load_state_dict(encoder_state_dict)

    ane_encoder = ANEEncoder(torch_encoder, max_seq_len=config.data.text_length)
    ane_encoder.layers = ane_encoder.layers[:2] # debug
    text = """
[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct!
[S2] Oh my god! Okay.. it's happening. Everybody stay calm!
[S1] What's the procedure...
[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! 
[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway!     
"""
    byte_text = text.encode("utf-8")
    replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
    text_tokens = list(replaced_bytes)
    encoded_text = torch.tensor(
        [text_tokens],
        dtype=torch.long,
    )
    encoded_text = torch.cat(
        (encoded_text, torch.zeros((1, 10), dtype=torch.long)),
        dim=1,
    )
    unconditional_text = torch.zeros_like(encoded_text)
    batch = torch.cat((unconditional_text, encoded_text), dim=0)

    import coremltools as ct
    import coremltools.converters.mil as mil
    from coremltools.converters.mil import Builder as mb
    import numpy as np

    input_lengths = [32, 64, 128, 256, 384, 512, 768, 1024]
    input_def = mil.input_types.EnumeratedShapes(shapes=[(2, s) for s in input_lengths])
    output_def = mil.input_types.EnumeratedShapes(shapes=[(2, torch_encoder.config.model.encoder.n_hidden, 1, s) for s in input_lengths])

    with torch.no_grad():
        # traced_model = torch.jit.trace(wmodel, example_inputs)
        traced_model = torch.jit.trace(ane_encoder, example_kwarg_inputs={"x_ids": batch})

    inputs = [
        ct.TensorType(name="x_ids", shape=input_def, dtype=np.int32),
    ]
    outputs = [
        # ct.TensorType(name="enc_out", shape=output_def, dtype=np.float16),
        ct.TensorType(name="enc_out", dtype=np.float16),
    ]

    mlmodel: ct.models.MLModel = ct.convert(
        traced_model,
        convert_to="milinternal",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        # pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT16,
    )
    print(mlmodel)
    mlmodel.export_as_multifunction = True
    mlmodel: ct.models.MLModel = ct.convert(
        mlmodel,
        # convert_to="milinternal",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        # compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        # pass_pipeline=pipeline,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(save_path)



if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    convert(args.model_path + "/config.json", args.model_path + "/model.safetensors", args.save_path)
