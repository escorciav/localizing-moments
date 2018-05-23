#!/bin/bash

echo "RGB model on val..."
python dump_features.py --deploy_net prototxts/deploy_clip_retrieval_rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt \
                       --snapshot_tag rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2 \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter 30000 \
                       --snapshot_interval 30000 \
                       --loc \
                       --test_h5 data/average_fc7.h5 \
                       --split val

echo "Flow model on val..."
python dump_features.py --deploy_net prototxts/deploy_clip_retrieval_flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt \
                       --snapshot_tag flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2 \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter 30000 \
                       --snapshot_interval 30000 \
                       --loc \
                       --test_h5 data/average_global_flow.h5 \
                       --split val

echo "RGB model on test..."
python dump_features.py --deploy_net prototxts/deploy_clip_retrieval_rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt \
                       --snapshot_tag rgb_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2 \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter 30000 \
                       --snapshot_interval 30000 \
                       --loc \
                       --test_h5 data/average_fc7.h5 \
                       --split test

echo "Flow model on test..."
python dump_features.py --deploy_net prototxts/deploy_clip_retrieval_flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2.prototxt \
                       --snapshot_tag flow_iccv_release_feature_process_context_recurrent_embedding_lfTrue_dv0.3_dl0.0_nlv2_nlllstm_no_embed_edl1000-100_edv500-100_pmFalse_losstriplet_lwInter0.2 \
                       --visual_feature feature_process_context \
                       --language_feature recurrent_embedding \
                       --max_iter 30000 \
                       --snapshot_interval 30000 \
                       --loc \
                       --test_h5 data/average_global_flow.h5 \
                       --split test
