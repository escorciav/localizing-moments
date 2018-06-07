import argparse

import caffe
import h5py


def extract_caffe_model(model, weights, filename):
    """Dump HDF5 with caffe's model's parameters

    Args:
      model: path of '.prototxt'
      weights: path of '.caffemodel'
      filename: output path of hdf5-file

    """
    net = caffe.Net(model, caffe.TEST, weights=weights)
    with h5py.File(filename, 'w') as fid:
        for item in net.params.items():
            name, layer = item
            print('convert layer: ' + name)
            num = 0
            for p in net.params[name]:
                fid.create_dataset(str(name) + '_' + str(num), data=p.data)
                num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="prototxt model")
    parser.add_argument("--weights", help="protobin model (.caffemodel)")
    parser.add_argument("--filename", help="output filename")
    args = parser.parse_args()
    extract_caffe_model(**vars(args))
