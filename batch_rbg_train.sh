num_runs=${1:-1}
offset=${2:-1}
for i in $(eval echo {$offset..$num_runs}); do
  python build_net.py  \
    --feature_process_visual feature_process_context \
    --loc \
    --vision_layers 2 \
    --lw_inter 0.2 \
    --dropout_visual 0.3 \
    --dropout_language 0.0 \
    --language_layers lstm_no_embed \
    --feature_process_language recurrent_embedding \
    --visual_embedding_dim 500 100 \
    --language_embedding_dim 1000 100 \
    --gpu 0 \
    --max_iter 30000 \
    --snapshot 10000 \
    --stepsize 10000 \
    --base_lr 0.05 \
    --random_seed 1701 \
    --tag rgb_hachiko_ \
    --exp-id $i
  echo
done