#!/bin/bash
max_iter=${3:-30000}
snapshot_interval=${4:-10000}
# shut-up caffe (comment it for debugging)
export GLOG_minloglevel=2

echo "Flow model on val..."
python test_network.py --deploy_net $1 \
                       --snapshot_tag $2 \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter $max_iter \
                       --snapshot_interval $snapshot_interval \
                       --loc \
                       --test_h5 data/average_global_flow.h5 \
                       --split val