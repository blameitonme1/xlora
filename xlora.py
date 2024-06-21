import torch
from torch import nn
import inspect

class xlora(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.non_linear = nn.ReLU()
        self.adapter_down = nn.Linear(768, dim, bias=False)
        self.adapter_up = nn.Linear(dim, 768, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
    
    def forward(self, x, weight):
        adapter_down = self.non_linear(weight @ self.adapter_down.weight.t()) # (W_0 W_1) [a, d]
        x = x @ adapter_down
        x = self.adapter_up(x)
        return x

def forward_attn(
        self, hidden_states, head_mask = None, output_attentions: bool = False
        ):
    mixed_query_layer = self.query(hidden_states) + self.q_adapter(hidden_states, self.query.weight)

    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states) + self.v_adapter(hidden_states, self.value.weight))
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
    
def set_xlora(model, dim):
    for name, layer in model.named_children():
        print(name)
        if 'attention' in name:
            print("------- set xlora in self attention -------")
            print(inspect.getsource(layer.attention.forward))
            layer.attention.q_adapter = xlora(dim=dim)
            layer.attention.v_adapter = xlora(dim=dim)
            bound_method = forward_attn.__get__(layer.attention, layer.attention.__class__)
            setattr(layer.attention, 'forward', bound_method)
            print(inspect.getsource(layer.attention.forward))
            print("-------     finish setting xlora    -------")
        elif len(list(layer.children())) != 0:
            set_xlora(layer, dim)