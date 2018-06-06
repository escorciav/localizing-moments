#!/bin/bash

rgb_prototxt=${1:-prototxts/deploy_clip_retrieval_rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt}
flow_prototxt=${2:-prototxts/deploy_clip_retrieval_flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt}
rgb_snapshot=${3:-non-existent}
flow_snapshot=${4:-non-existent}

# shut-up caffe (comment it for debugging)
export GLOG_minloglevel=2
echo "RGB model on val..."
python dump_features.py --deploy_net $rgb_prototxt \
                       --snapshot $rgb_snapshot \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --loc \
                       --test_h5 data/average_fc7.h5 \
                       --split val

echo "Flow model on val..."
python dump_features.py --deploy_net $flow_prototxt \
                       --snapshot $flow_snapshot \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --loc \
                       --test_h5 data/average_global_flow.h5 \
                       --split val

echo "RGB model on test..."
python dump_features.py --deploy_net $rgb_prototxt \
                       --snapshot $rgb_snapshot \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --loc \
                       --test_h5 data/average_fc7.h5 \
                       --split test

echo "Flow model on test..."
python dump_features.py --deploy_net $flow_prototxt \
                       --snapshot $flow_snapshot \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --loc \
                       --test_h5 data/average_global_flow.h5 \
                       --split test
