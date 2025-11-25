import torch
import os
import sys
from collections import OrderedDict
from einops import rearrange
def create_new_state_dict(state_dict, num_layers, i2v=False, vace_layers=None):
    i2v = False
    # hidden_size = 1536 # 5120 for 14b
    # num_layers = 30
    # num_attention_heads = 12

    sat_weight = {}

    wan_weight = {}
    for k, v in state_dict.items():
        if k.startswith('model.diffusion_model.'):
            wan_weight[k.removeprefix('model.diffusion_model.')] = v
        else:
            wan_weight[k] = v
    print(wan_weight['blocks.1.self_attn.norm_q.weight'].shape)
    # wan_weight = state_dict
    # return

    used_keys = set()

    def use_key(key):
        used_keys.add(key)
        return wan_weight[key]

    sat_weight['model.diffusion_model.mixins.patch_embed.proj.weight'] = use_key('patch_embedding.weight')
    sat_weight['model.diffusion_model.mixins.patch_embed.proj.bias'] = use_key('patch_embedding.bias')

    sat_weight['model.diffusion_model.text_embedding.0.weight'] = use_key('text_embedding.0.weight')
    sat_weight['model.diffusion_model.text_embedding.0.bias'] = use_key('text_embedding.0.bias')

    sat_weight['model.diffusion_model.text_embedding.2.weight'] = use_key('text_embedding.2.weight')
    sat_weight['model.diffusion_model.text_embedding.2.bias'] = use_key('text_embedding.2.bias')

    sat_weight['model.diffusion_model.mixins.final_layer.adaLN_modulation'] = use_key('head.modulation')
    sat_weight[f'model.diffusion_model.mixins.final_layer.linear.weight'] = use_key(f'head.head.weight')
    sat_weight[f'model.diffusion_model.mixins.final_layer.linear.bias'] = use_key(f'head.head.bias')

    sat_weight[f'model.diffusion_model.time_embed.0.weight'] = use_key('time_embedding.0.weight')
    sat_weight[f'model.diffusion_model.time_embed.0.bias'] = use_key('time_embedding.0.bias')
    sat_weight['model.diffusion_model.time_embed.2.weight'] = use_key('time_embedding.2.weight')
    sat_weight['model.diffusion_model.time_embed.2.bias'] = use_key('time_embedding.2.bias')

    sat_weight['model.diffusion_model.adaln_projection.1.weight'] = use_key('time_projection.1.weight')
    sat_weight['model.diffusion_model.adaln_projection.1.bias'] = use_key('time_projection.1.bias')

    if i2v:
        sat_weight['model.diffusion_model.clip_proj.proj.0.weight'] = use_key('img_emb.proj.0.weight')
        sat_weight['model.diffusion_model.clip_proj.proj.0.bias'] = use_key('img_emb.proj.0.bias')
        sat_weight['model.diffusion_model.clip_proj.proj.1.weight'] = use_key('img_emb.proj.1.weight')
        sat_weight['model.diffusion_model.clip_proj.proj.1.bias'] = use_key('img_emb.proj.1.bias')
        sat_weight['model.diffusion_model.clip_proj.proj.3.weight'] = use_key('img_emb.proj.3.weight')
        sat_weight['model.diffusion_model.clip_proj.proj.3.bias'] = use_key('img_emb.proj.3.bias')
        sat_weight['model.diffusion_model.clip_proj.proj.4.weight'] = use_key('img_emb.proj.4.weight')
        sat_weight['model.diffusion_model.clip_proj.proj.4.bias'] = use_key('img_emb.proj.4.bias')

    for i in range(num_layers):
        # self attn
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.attention.dense.weight'] = use_key(f'blocks.{i}.self_attn.o.weight')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.attention.dense.bias'] = use_key(f'blocks.{i}.self_attn.o.bias')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.attention.query_key_value.weight'] = \
            torch.cat([
                use_key(f'blocks.{i}.self_attn.q.weight'),
                use_key(f'blocks.{i}.self_attn.k.weight'),
                use_key(f'blocks.{i}.self_attn.v.weight')
            ], dim=0)
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.attention.query_key_value.bias'] = \
            torch.cat([
                use_key(f'blocks.{i}.self_attn.q.bias'),
                use_key(f'blocks.{i}.self_attn.k.bias'),
                use_key(f'blocks.{i}.self_attn.v.bias')
            ], dim=0)

        sat_weight[f'model.diffusion_model.mixins.adaln_layer.query_layernorm_list.{i}.weight'] = use_key(f'blocks.{i}.self_attn.norm_q.weight')
        sat_weight[f'model.diffusion_model.mixins.adaln_layer.key_layernorm_list.{i}.weight'] = use_key(f'blocks.{i}.self_attn.norm_k.weight')

        # mlp
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.mlp.dense_h_to_4h.weight'] = use_key(f'blocks.{i}.ffn.0.weight')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.mlp.dense_h_to_4h.bias'] = use_key(f'blocks.{i}.ffn.0.bias')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.mlp.dense_4h_to_h.weight'] = use_key(f'blocks.{i}.ffn.2.weight')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.mlp.dense_4h_to_h.bias'] = use_key(f'blocks.{i}.ffn.2.bias')

        # cross attn
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.cross_attention.dense.weight'] = use_key(f'blocks.{i}.cross_attn.o.weight')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.cross_attention.dense.bias'] = use_key(f'blocks.{i}.cross_attn.o.bias')

        sat_weight[f'model.diffusion_model.transformer.layers.{i}.cross_attention.query.weight'] = use_key(f'blocks.{i}.cross_attn.q.weight')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.cross_attention.query.bias'] = use_key(f'blocks.{i}.cross_attn.q.bias')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.cross_attention.key_value.weight'] = torch.cat([
            use_key(f'blocks.{i}.cross_attn.k.weight'),
            use_key(f'blocks.{i}.cross_attn.v.weight'),
        ], dim=0)
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.cross_attention.key_value.bias'] = torch.cat([
            use_key(f'blocks.{i}.cross_attn.k.bias'),
            use_key(f'blocks.{i}.cross_attn.v.bias')
        ], dim=0)

        sat_weight[f'model.diffusion_model.transformer.layers.{i}.post_cross_attention_layernorm.weight'] = use_key(f'blocks.{i}.norm3.weight')
        sat_weight[f'model.diffusion_model.transformer.layers.{i}.post_cross_attention_layernorm.bias'] = use_key(f'blocks.{i}.norm3.bias')

        sat_weight[f'model.diffusion_model.mixins.adaln_layer.cross_query_layernorm_list.{i}.weight'] = use_key(f'blocks.{i}.cross_attn.norm_q.weight')
        sat_weight[f'model.diffusion_model.mixins.adaln_layer.cross_key_layernorm_list.{i}.weight'] = use_key(f'blocks.{i}.cross_attn.norm_k.weight')

        if i2v:
            sat_weight[f'model.diffusion_model.mixins.adaln_layer.clip_feature_key_layernorm_list.{i}.weight'] = use_key(f'blocks.{i}.cross_attn.norm_k_img.weight')
            sat_weight[f'model.diffusion_model.mixins.adaln_layer.clip_feature_key_value_list.{i}.weight'] = torch.cat([
                use_key(f'blocks.{i}.cross_attn.k_img.weight'),
                use_key(f'blocks.{i}.cross_attn.v_img.weight'),
            ], dim=0)
            sat_weight[f'model.diffusion_model.mixins.adaln_layer.clip_feature_key_value_list.{i}.bias'] = torch.cat([
                use_key(f'blocks.{i}.cross_attn.k_img.bias'),
                use_key(f'blocks.{i}.cross_attn.v_img.bias'),
            ], dim=0)

        # adaln
        sat_weight[f'model.diffusion_model.mixins.adaln_layer.adaLN_modulations.{i}'] = use_key(f'blocks.{i}.modulation')

    unused_keys = set(wan_weight.keys()) - used_keys
    if unused_keys:
        print("Unused keys in wan_weight:")
        for key in sorted(unused_keys):
            print(" ", key)
    else:
        print("All keys in wan_weight were used.")

    return {
        'module': OrderedDict(sat_weight)
    }


def main_1_3b():
    from safetensors.torch import load_file

    checkpoint_path = '1B_wan'
    state_dict = load_file(checkpoint_path)

    state = create_new_state_dict(state_dict, num_layers=30)
    
    new_dir = './ckpt/wan_1B_sat'
    os.makedirs(f"{new_dir}/1", exist_ok=True)
    
    torch.save(state, f"{new_dir}/1/mp_rank_00_model_states.pt")
    
    with open(f'{new_dir}/latest', 'w') as f:
        f.write('1')


def main_14B():
    from safetensors.torch import load_file
    import json

    checkpoint_dir = '14B_wan'
    state_dict = {}
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.safetensors'):
            checkpoint_path = os.path.join(checkpoint_dir, file)
            state_dict.update(load_file(checkpoint_path))
    num_layers = 40
    state = create_new_state_dict(state_dict, num_layers)
    new_dir = './ckpt/wan_14B_sat'
    os.makedirs(f"{new_dir}/1", exist_ok=True)
    
    torch.save(state, f"{new_dir}/1/mp_rank_00_model_states.pt")
    
    with open(f'{new_dir}/latest', 'w') as f:
        f.write('1')

if __name__ == "__main__":
    main_14B()