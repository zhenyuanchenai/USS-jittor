import jittor as jt
from jittor import nn, Module
from typing import List, Tuple, Type
from .common import LayerNorm2d

class MaskDecoder(Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: Module,
        num_multimask_outputs: int = 3,
        activation: Type[Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1  
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def execute(
        self,
        image_embeddings: jt.Var,
        image_pe: jt.Var,
        sparse_prompt_embeddings: jt.Var,
        dense_prompt_embeddings: jt.Var,
        multimask_output: bool,
    ) -> Tuple[jt.Var, jt.Var]:

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: jt.Var,
        image_pe: jt.Var,
        sparse_prompt_embeddings: jt.Var,
        dense_prompt_embeddings: jt.Var,
    ) -> Tuple[jt.Var, jt.Var]:
        
        output_tokens = jt.concat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = jt.concat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings.repeat(tokens.shape[0], 1, 1, 1)
        src = src + dense_prompt_embeddings 
        pos_src = image_pe.repeat(tokens.shape[0], 1, 1, 1)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1,2).reshape(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[jt.Var] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = jt.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )
        self.sigmoid_output = sigmoid_output

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = nn.Sigmoid(x)
        return x
        