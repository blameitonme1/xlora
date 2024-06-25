import torch
from torch import nn
import inspect
import torch.nn.functional as F

class vanilla_xlora(nn.Module):
    """
    vanilla implementation of xLoRA, no residual, no depth
    use ReLU as the default non-linear function
    """
    def __init__(self, dim=32, out_dim=768):
        super().__init__()
        self.non_linear = nn.ReLU()
        self.adapter_down = nn.Linear(out_dim, dim, bias=False)
        self.adapter_up = nn.Linear(dim, out_dim, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
    
    def forward(self, x, weight):
        adapter_down = self.non_linear(weight @ self.adapter_down.weight.t()) # (W_0 W_1) [a, d]
        x = x @ adapter_down
        x = self.adapter_up(x)
        return x

def forward_attn(
        self, hidden_states, head_mask = None, output_attentions: bool = False
        ):
    mixed_query_layer = self.query(hidden_states) + 0.01 * self.q_adapter(hidden_states, self.query.weight.t())

    key_layer = self.transpose_for_scores(self.key(hidden_states) + (0.01 * self.k_adapter(hidden_states, self.key.weight.t()) if self.xlora_mode == 2 or self.xlora_mode == 5 else 0))
    value_layer = self.transpose_for_scores(self.value(hidden_states) + 0.01 * self.v_adapter(hidden_states, self.value.weight.t()))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer,
        key_layer,
        value_layer,
        head_mask,
        self.attention_probs_dropout_prob if self.training else 0.0,
        is_causal=False,
        scale=None,
    )

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    return context_layer, None

def forward_ffn(self, input):
        return F.linear(input, self.weight, self.bias) + 0.01 * self.adapter(input, self.weight.t())
 
def set_xlora(model, xlora_mode=1, mhsa_dim=16, ffn_dim=16):
    """
    xlora mode:
        1: vanilla xlora on mhsa q, v
        2: vanilla xlora on mhsa q, k, v
        3: vanilla xlora on ffn
        4: vanilla xlora on both mhsa q, v and ffn
        5: vanilla xlora on both mhsa q, k, v and ffn
    """
    for name, layer in model.named_children():
        print(name)
        if 'attention' in name:
            if xlora_mode == 3:
                continue
            print("------- set xlora in self attention -------")
            layer.attention.xlora_mode = xlora_mode
            # print(inspect.getsource(layer.attention.forward))
            layer.attention.q_adapter = vanilla_xlora(dim=mhsa_dim)
            layer.attention.v_adapter = vanilla_xlora(dim=mhsa_dim)
            if xlora_mode == 2 or xlora_mode == 5:
                layer.attention.k_adapter = vanilla_xlora(dim=mhsa_dim)
            bound_method = forward_attn.__get__(layer.attention, layer.attention.__class__)
            setattr(layer.attention, 'forward', bound_method)
            # print(inspect.getsource(layer.attention.forward))
            # print("-------     finish setting xlora    -------")
        elif 'dense' in name:
            print(layer.weight.shape)
            if xlora_mode == 1 or xlora_mode == 2:
                continue
            print("------- set xlora in FFN    -------")
            # print(inspect.getsource(layer.forward))
            layer.xlora_mode = xlora_mode
            layer.adapter = vanilla_xlora(dim=ffn_dim, out_dim=layer.weight.shape[0])
            bound_method = forward_ffn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            # print("------- finish setting xlora-------")
        elif len(list(layer.children())) != 0:
            set_xlora(layer, xlora_mode, mhsa_dim, ffn_dim)