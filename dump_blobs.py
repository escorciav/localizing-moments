import sys
import os
os.environ['GLOG_minloglevel'] = '2'  # suppress log / decrease for debugging
sys.path.append('utils/')
from config import *
sys.path.append(pycaffe_dir)
import caffe
from utils import *
from data_processing import *
from eval import *
import numpy as np
import pickle as pkl
import copy
import argparse
from collections import OrderedDict
import random
caffe.set_mode_gpu()
caffe.set_device(device_id)
MAX_NUMBER_SAMPLES = 5


def test_model(deploy_net, snapshot,
               visual_feature='feature_process_norm',
               language_feature='recurrent_embedding',
               loc=False,
               test_h5='data/average_fc7.h5',
               split='val'):

    params = {'feature_process': visual_feature, 'loc_feature': loc, 'loss_type': 'triplet',
              'batch_size': 120, 'features': test_h5, 'oversample': False, 'sentence_length': 50,
              'query_key': 'query', 'cont_key': 'cont', 'feature_key_p': 'features_p',
              'feature_time_stamp_p': 'feature_time_stamp_p',
              'feature_time_stamp_n': 'feature_time_stampe_n'}

    language_extractor_fcn = extractLanguageFeatures
    visual_extractor_fcn = extractVisualFeatures

    language_process = language_feature_process_dict[language_feature]
    data_orig = read_json('data/%s_data.json' %split)
    language_processor = language_process(data_orig)
    data = language_processor.preprocess(data_orig)
    params['vocab_dict'] = language_processor.vocab_dict
    num_glove_centroids = language_processor.get_vector_dim()
    params['num_glove_centroids'] = num_glove_centroids
    thread_result = {}

    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations

    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations

    sorted_segments_list = []
    net = caffe.Net(deploy_net, snapshot, caffe.TEST)
    data_blobs = {}
    random.shuffle(data)

    for id, d in enumerate(data[:MAX_NUMBER_SAMPLES]):
        video_id = d['video']
        query_id = d['annotation_id']

        vis_features, loc_features = visual_feature_extractor.get_data_test({'video': video_id})
        lang_features, cont = textual_feature_extractor.get_data_test(d)

        net.blobs['image_data'].data[...] = vis_features.copy()
        net.blobs['loc_data'].data[...] = loc_features.copy()

        for i in range(vis_features.shape[0]):
            net.blobs['text_data'].data[:,i,:] = lang_features
            net.blobs['cont_data'].data[:,i] = cont

        net.forward()
        data_blobs[query_id] = {}
        for k, v in net.blobs.iteritems():
            data_blobs[query_id][k] = v.data.copy()

    print 'Dumping features'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # TODO edit this shit
    filename = os.path.join(result_dir,
                            'sample_blobs_{}_{}.hdf5'
                            .format(os.path.basename(snapshot), split))
    with h5py.File(filename, 'w') as fid:
        for k, v in data_blobs.iteritems():
            grp = fid.create_group(str(k))
            for blob_name, blob_data in v.iteritems(): 
                grp.create_dataset(blob_name, data=blob_data, chunks=True)
    print "Dumped results to: {}".format(filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--deploy_net", type=str, default=None)
    parser.add_argument("--snapshot", type=str, default=None)
    parser.add_argument("--visual_feature", type=str, default="feature_process_norm")
    parser.add_argument("--language_feature", type=str, default="recurrent_embedding")
    parser.add_argument("--loc", dest='loc', action='store_true')
    parser.set_defaults(loc=False)
    parser.add_argument("--test_h5", type=str, default='data/average_fc7.h5')
    parser.add_argument("--split", type=str, default='val')

    args = parser.parse_args()

    test_model(args.deploy_net, args.snapshot,
               visual_feature = args.visual_feature,
               language_feature = args.language_feature,
               loc = args.loc,
               test_h5 = args.test_h5,
               split = args.split)
