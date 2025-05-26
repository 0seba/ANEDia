from math import isnan
import time

import torch
import torch.nn as nn
import numpy as np

import coremltools as ct
from tqdm import tqdm


class FlexState(nn.Module):
    def __init__(self, nlayers=4):
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(nlayers):
            conv = nn.Conv2d(
                in_channels=256, out_channels=64 * 4, kernel_size=1, bias=False
            )
            conv.weight.data.normal_(std=0.1)
            self.convs.append(conv)

        kv = torch.randn((nlayers, 4, 1024, 64)) * 0.1
        self.register_buffer("kv", kv)

    def forward(self, x: torch.Tensor, length: torch.Tensor):
        seqlen = x.shape[-1]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = x.view(x.shape[0], 4, 64, seqlen)
            x = x.permute(0, 1, 3, 2)
            kv = self.kv[i : i + 1, :, 0 : length.shape[0]]
            print(x.size(), kv.size())
            x = torch.nn.functional.scaled_dot_product_attention(x, kv, kv)
            print(x.size())
            x = x.permute(0, 1, 3, 2)
            x = x.view(x.shape[0], 256, 1, seqlen)

        return x

    def convert(self):
        import coremltools as ct

        with torch.inference_mode():
            example_input = torch.randn((1, 256, 1, 1)) * 0.1
            example_length = torch.zeros((64,), dtype=torch.int32)
            traced_model = torch.jit.trace(self, (example_input, example_length))

        inputs = [
            ct.TensorType(
                name="x",
                shape=(1, 256, 1, 1),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="length",
                shape=ct.EnumeratedShapes(
                    [(s,) for s in [32, 64, 128, 256, 384, 512, 768, 1024]]
                ),
                dtype=np.int32,
            ),
        ]
        outputs = [
            ct.TensorType(
                name="output",
                dtype=np.float16,
            ),
        ]
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=self.kv.size()),
                name="kv",
            ),
        ]

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
        mlmodel = ct.convert(
            mlmodel,
            # convert_to="milinternal",
            inputs=inputs,
            outputs=outputs,
            # states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=False,
        )
        mlmodel.save("flex_state")


def test_generation_speed():
    mlmodels = []
    for path in [
        "dia_decoder_1_6.mlmodelc",
        # "dia_decoder_7_12.mlmodelc",
        # "dia_decoder_13_18.mlmodelc",
    ]:
        mlmodels.append(
            ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        )
    cross_attention_model = ct.models.CompiledMLModel(
        "precompute_cross_attn_cach.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE
    )

    state = cross_attention_model.make_state()
    enc_out = np.random.normal(scale=0.1, size=(2, 1024, 1, 512))  # .astype(np.float16)
    cross_attention_model.predict({"enc_out": enc_out}, state=state)

    dec_in = np.random.normal(scale=0.1, size=(2, 2048, 1, 1))
    enc_lengths = np.array([512, 512], dtype=np.int32)
    attn_mask = np.arange(2048, dtype=np.int32)
    attn_mask = attn_mask[:, None] <= attn_mask[None, :]
    attn_mask = np.where(
        attn_mask, np.array(0.0, dtype=np.float16), -np.array(np.inf, dtype=np.float16)
    )
    attn_mask = attn_mask[None, None, :, :].repeat(2, 0)
    n = 1000
    times = []
    for i in tqdm(range(n)):
        attn_mask = attn_mask[:, :, i : i + 1]
        positions = np.array([[i], [i]], dtype=np.int32)
        kv_write_index = np.array([i], dtype=np.int32)
        start_time = time.time()

        for mlmodel in mlmodels:
            dec_in = mlmodel.predict(
                {
                    "x": dec_in,
                    "encoder_lengths": enc_lengths,
                    "self_attn_mask": attn_mask,
                    "positions": positions,
                    "kv_write_index": kv_write_index,
                },
                state=state,
            )
            print(dec_in)
            dec_in = dec_in["hidden_states"]

        times.append(time.time() - start_time)
        if np.isnan(dec_in):
            print("nannananananananana")
    print(times[5:] / (n - 5))
    print(times[:30])


if __name__ == "__main__":
    # FlexState().convert()
    test_generation_speed()
