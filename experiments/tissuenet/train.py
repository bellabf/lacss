#!/usr/bin/env python

from logging.config import valid_ident
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import json
from functools import partial
import argparse
from os.path import join
import numpy as np
import jax
import optax
import elegy as eg
import tensorflow as tf
from skimage.measure import regionprops
from tqdm import tqdm

jnp = jax.numpy

import lacss

tf.config.set_visible_devices([], 'GPU')

# global defaults
batch_size = 1
learning_rate = 2e-3 * batch_size
training_epochs = 16
steps_per_epoch = 2601 // batch_size

platform_names = ['codex', 'cycif', 'imc', 'mibi', 'mxif', 'vectra']
tissue_names = ['breast', 'gi', 'immune', 'lung', 'pancreas', 'skin']

def tissue_net_gen_fn(data_path):
    X = np.load(join(data_path, 'X.npy'), mmap_mode='r+')
    Y = np.load(join(data_path, 'y.npy'), mmap_mode='r+')
    platforms = np.load(join(data_path, 'platform_list.npy'))
    tissues = np.load(join(data_path, 'tissue_list.npy'))
    for x, y, pf, t in zip(X, Y, platforms, tissues):
        img = x.astype('float32')
        img = np.pad(img, [[0,0],[0,0],[0,1]]) # make 3-ch
        label_in_ch0 = np.argmax(np.count_nonzero(y, axis=(0,1))) == 0
        y = y[..., 0] if label_in_ch0 else y[..., 1]
        binary_mask = (y > 0).astype('float32')
        locs = [prop['centroid'] for prop in regionprops(y)]
        bboxes = []
        for prop in regionprops(y):
            bboxes.append(prop['bbox'])
        
        bboxes = np.array(bboxes, dtype='float32')

        yield {
            'image': img,
            'locations': locs,
            'binary_mask': binary_mask,
            'bboxes': bboxes,
            'mask_labels': y,
            'platform': platform_names.index(pf),
            'tissue': tissue_names.index(t),
        }

def train_parser(x):
    pf = x['platform']
    x = lacss.data.parse_train_data_func(x, size_jitter=(0.85, 1.15), target_height=544, target_width=544)    
    x_data = dict(
        image = x['image'],
        gt_locations = x['locations'],
    )
    y_data = dict(
        group_num = tf.cast(pf, tf.float32),
        binary_mask = tf.cast(x['binary_mask'], tf.float32),
    )
    return x_data, y_data

def prepare_data(n_buckets=8):
    output_signiture =  {
            'image': tf.TensorSpec([None, None, 3], dtype=tf.float32),
            'locations': tf.TensorSpec([None, 2], dtype=tf.float32),
            'binary_mask': tf.TensorSpec([None, None], dtype=tf.float32),
            'bboxes': tf.TensorSpec([None, 4], dtype=tf.float32),
            'mask_labels': tf.TensorSpec([None, None], tf.float32),
            'platform': tf.TensorSpec([], tf.int32),
            'tissue': tf.TensorSpec([], tf.int32),
        }

    ds_train = tf.data.Dataset.from_generator(
        lambda: tissue_net_gen_fn(join(args.datapath, 'train')),
        output_signature = output_signiture
    ).cache(join(args.logpath, 'train')).map(train_parser).repeat()

    ds_train = ds_train.bucket_by_sequence_length(
        element_length_func = lambda x, y: tf.shape(x['gt_locations'])[0],
        bucket_boundaries = list(np.arange(1, n_buckets) * 4096 // n_buckets + 1),
        bucket_batch_sizes = (batch_size,) * n_buckets,
        padding_values = -1.0,
        pad_to_bucket_boundary = True,
    )

    ds_val = tf.data.Dataset.from_generator(
        lambda: tissue_net_gen_fn(join(args.datapath, 'val')),
        output_signature = output_signiture
    ).cache(join(args.logpath, 'val'))

    return ds_train, ds_val

def pad_to(x, multiple=256):
    s = x.shape[0]
    ns = ((s - 1) // multiple + 1) * multiple
    padding = ns - s
    return np.pad(x, [[0,padding],[0,0]]), s

def cb_fn(epoch, logs, model, ds):
    model = model.eval()
    metrics = eg.metrics.Metrics([
        lacss.metrics.LoiAP([0.1, 0.2, 0.5, 1.0]),
        lacss.metrics.BoxAP([0.5, 0.75])
    ])
    
    for data in ds.skip(3).take(3):
        del data['mask_indices']
        preds = model.predict_on_batch(jnp.array(data['image']))
        labels = dict(
            gt_locations = jnp.array(data['locations']),
            gt_boxes = jnp.array(data['bboxes']),
        )
        metrics.update(preds=preds, **labels)
        
    logs.update(metrics.compute())

def run_training():
    ds_train, ds_val = prepare_data()
    print(ds_train.element_spec)

    if args.resume != "":
        model = eg.model.model_base.load(args.resume)
        print(f'Loaded checkpoint {args.resume}')
        init_epoch = int(args.resume.split('-')[-1])
    else:
        if args.config != "":
            with open(args.config) as f:
                model_cfg = json.load(f)
        else:
            model_cfg = dict(
                detector = dict(
                    train_pre_nms_topk = 2048,
                    train_max_output = 1024,
                    train_min_score = 0.4,
                    test_pre_nms_topk = -1,
                    test_max_output = 1024,
                    test_min_score = 0.2,
                ),
                auxnet = dict(
                    n_groups=6,
                )
            )
        module = lacss.modules.Lacss.from_config(model_cfg)

        optimizer = optax.adamw(0.0005)

        loss = [
            lacss.losses.DetectionLoss(),
            lacss.losses.LocalizationLoss(),
            lacss.losses.InstanceEdgeLoss(),
            lacss.losses.InstanceLoss(),
        ]

        model = eg.Model(
            module = module,     
            optimizer = optimizer,
            seed = args.seed,
            loss = loss,
        )
        init_epoch = 0

    with open(join(args.logpath, 'config.json'), 'w') as f:
        json.dump(model.module.get_config(), f)

    model.fit(
        inputs=ds_train, 
        epochs=training_epochs, 
        steps_per_epoch=steps_per_epoch, 
        initial_epoch=init_epoch,
        verbose = args.verbose,
        callbacks = [
            eg.callbacks.TensorBoard(args.logpath),
            eg.callbacks.ModelCheckpoint(path=join(args.logpath, 'chkpt-{epoch:02d}')),
            eg.callbacks.LambdaCallback(on_epoch_end = partial(cb_fn, model=model, ds=ds_val)),
        ]
    )

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train livecell model')
    parser.add_argument('datapath', type=str, help='Data dir of tfrecord files')
    parser.add_argument('logpath', type=str, help='Log dir for storing results')
    parser.add_argument('--config', type=str, default='', help='path to the model config file')   
    # parser.add_argument('--supervised', type=bool, default=False, help='Whether train superversed')
    parser.add_argument('--resume', type=str, default="", help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--verbose', type=int, default=2, help='output verbosity')

    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    try:
        os.makedirs(args.logpath)
    except:
        pass

    run_training()
