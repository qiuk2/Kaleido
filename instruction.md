# download model

```
# 3) Download the whole repo to a local folder
huggingface-cli download zai-org/Kaleido-14B-S2V \
  --local-dir Kaleido-14B-S2V \
  --local-dir-use-symlinks False

cd Kaleido-14B-S2V

# Merge the checkpoint files
python merge_kaleido.py

```

# additional package needs to download
```
pip install omegaconf einops pytorch_lightning scipy beartype deepspeed tensorboardX datasets ftfy
```

# Running code

Change the image_root and output_dir in configs/sampling/sample_wanvae_concat_14b.yaml

```
torchrun \
    --nnodes $MLP_WORKER_NUM \
    --nproc_per_node $MLP_GPU \
    --node_rank $NODE_RANK \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    sample_video.py \
    --base configs/video_model/dit_crossattn_14B_wanvae.yaml \
    configs/sampling/sample_wanvae_concat_14b.yaml
```
