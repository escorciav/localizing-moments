import sys
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
import os
caffe.set_mode_gpu()
caffe.set_device(device_id)

L2_TEST_DECIMAL = 6
L2_TEST_EXHAUSTIVITY = 0.95  # less than zero to run test over entire set
os.environ['GLOG_minloglevel'] = '2'  # suppress log / decrease for debugging

def test_model(deploy_net, snapshot_tag,
               visual_feature='feature_process_norm',
               language_feature='recurrent_embedding',
               max_iter=30000,
               snapshot_interval=30000,
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

    snapshot = '%s/%s_iter_%%d.caffemodel' %(snapshot_dir, snapshot_tag)

    visual_feature_extractor = visual_extractor_fcn(data, params, thread_result)
    textual_feature_extractor = language_extractor_fcn(data, params, thread_result)
    possible_segments = visual_feature_extractor.possible_annotations

    assert snapshot_interval == max_iter
    sorted_segments_list = []
    net = caffe.Net(deploy_net, snapshot % snapshot_interval, caffe.TEST)
    queries, video_corpus, scores = OrderedDict(), OrderedDict(), OrderedDict()
    top_name = 'rank_score'

    for id, d in enumerate(data):
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

        if video_id not in video_corpus:
            video_corpus[video_id] = net.blobs['embedding_visual'].data.copy()

        queries[query_id] = net.blobs['embedding_text'].data[0, :].copy()
        scores[query_id] = net.blobs[top_name].data.copy()

        # toss a coin to check if l2-distance match
        if np.random.rand() > L2_TEST_EXHAUSTIVITY:
            np.testing.assert_almost_equal(
                ((video_corpus[video_id] - queries[query_id])**2).sum(axis=1),
                scores[query_id],
                decimal=L2_TEST_DECIMAL)

        if id % 10 == 0:
            sys.stdout.write('\r%d/%d' %(id, len(data)))

    print 'Dumping features'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    cue = ''
    if 'flow' in test_h5:
        cue = '_flow'

    filename = os.path.join(result_dir,
                            'queries_{}{}.hdf5'.format(split, cue))

    with h5py.File(filename, 'w') as fid:
        for k, v in queries.iteritems():
            k = str(k)
            fid.create_dataset(k, data=v, chunks=True)
    print "Dumped results to: {}".format(filename)

    filename = os.path.join(result_dir,
                            'corpus_{}{}.hdf5'.format(split, cue))
    with h5py.File(filename, 'w') as fid:
        for k, v in video_corpus.iteritems():
            fid.create_dataset(k, data=v, chunks=True)
    print "Dumped results to: {}".format(filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--deploy_net", type=str, default=None)
    parser.add_argument("--snapshot_tag", type=str, default=None)
    parser.add_argument("--visual_feature", type=str, default="feature_process_norm")
    parser.add_argument("--language_feature", type=str, default="recurrent_embedding")
    parser.add_argument("--max_iter", type=int, default=30000)
    parser.add_argument("--snapshot_interval", type=int, default=30000)
    parser.add_argument("--loc", dest='loc', action='store_true')
    parser.set_defaults(loc=False)
    parser.add_argument("--test_h5", type=str, default='data/average_fc7.h5')
    parser.add_argument("--split", type=str, default='val')

    args = parser.parse_args()

    test_model(args.deploy_net, args.snapshot_tag,
               visual_feature = args.visual_feature,
               language_feature = args.language_feature,
               max_iter = args.max_iter,
               snapshot_interval = args.snapshot_interval,
               loc = args.loc,
               test_h5 = args.test_h5,
               split = args.split)
