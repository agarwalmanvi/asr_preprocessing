#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make phone-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, basename
from tqdm import tqdm

import numpy as np


class Phone2idx(object):
    """Convert from phone to index.
    Args:
        vocab_file_path (string): path to the vocabulary file
        remove_list (list, optional): phones to neglect
    """

    def __init__(self, vocab_file_path, remove_list=[]):
        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                phone = line.strip()
                if phone in remove_list:
                    continue
                self.map_dict[phone] = vocab_count
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict['<'] = vocab_count
        self.map_dict['>'] = vocab_count + 1

    def __call__(self, str_phone):
        """
        Args:
            str_phone (string): string of space-divided phones
        Returns:
            index_list (np.ndarray): phone indices
        """
        # Convert from phone to the corresponding indices
        phone_list = str_phone.split(' ')
        index_list = list(map(lambda x: self.map_dict[x], phone_list))

        return np.array(index_list)


import os


def mkdir(path_to_dir):
    """Make a new directory if the directory does not exist.
    Args:
        path_to_dir (string): path to a directory
    Returns:
        path (string): path to the new directory
    """
    if path_to_dir is not None and (not os.path.isdir(path_to_dir)):
        os.mkdir(path_to_dir)
    return path_to_dir


def mkdir_join(path_to_dir, *dir_name):
    """Concatenate root path and 1 or more paths, and make a new direcory if
    the direcory does not exist.
    Args:
        path_to_dir (string): path to a diretcory
        dir_name (string): a direcory name
    Returns:
        path to the new directory
    """
    if path_to_dir is None:
        return path_to_dir

    path_to_dir = mkdir(path_to_dir)
    for i in range(len(dir_name)):
        if i == len(dir_name) - 1 and '.' in dir_name[i]:
            path_to_dir = os.path.join(path_to_dir, dir_name[i])
        else:
            path_to_dir = mkdir(os.path.join(path_to_dir, dir_name[i]))
    return path_to_dir


def map_phone2phone(phone_list, label_type, map_file_path):
    """Map from 61 phones to 39 or 48 phones.
    Args:
        phone_list (list): list of 61 phones (string)
        label_type (string): phone39 or phone48 or phone61
        map_file_path (string): path to the phone2phone mapping file
    Returns:
        mapped_phone_list (list): list of phones (string)
    """
    if label_type == 'phone61':
        return phone_list

    # read a mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[1] != 'nan':
                if label_type == 'phone48':
                    map_dict[line[0]] = line[1]
                elif label_type == 'phone39':
                    map_dict[line[0]] = line[2]
            else:
                map_dict[line[0]] = ''

    # mapping from 61 phones to 39 or 48 phones
    mapped_phone_list = []
    for i in range(len(phone_list)):
        if phone_list[i] in map_dict.keys():
            mapped_phone_list.append(map_dict[phone_list[i]])
        else:
            mapped_phone_list.append(phone_list[i])

    # ignore "q"
    while '' in mapped_phone_list:
        mapped_phone_list.remove('')

    return mapped_phone_list


def read_phone(label_paths, vocab_file_save_path, save_vocab_file=False,
               is_test=False):
    """Read phone transcript.
    Args:
        label_paths (list): list of paths to label files
        vocab_file_save_path (string): path to vocabulary files
        save_vocab_file (bool, optional): if True, save vocabulary files
        is_test (bool, optional): set True in case of the test set
    Returns:
        text_dict (dict):
            key (string) => utterance name
            value (list) => list of [phone61_indices, phone48_indices, phone39_indices]
    """
    print('=====> Reading target labels...')

    # Make the mapping file (from phone to index)
    phone2phone_map_file_path = join(
        vocab_file_save_path, '../phone2phone.txt')
    phone61_set, phone48_set, phone39_set = set([]), set([]), set([])
    with open(phone2phone_map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[1] != 'nan':
                phone61_set.add(line[0])
                phone48_set.add(line[1])
                phone39_set.add(line[2])
            else:
                # Ignore "q" if phone39 or phone48
                phone61_set.add(line[0])

    phone61_vocab_map_file_path = mkdir_join(
        vocab_file_save_path, 'phone61.txt')
    phone48_vocab_map_file_path = mkdir_join(
        vocab_file_save_path, 'phone48.txt')
    phone39_vocab_map_file_path = mkdir_join(
        vocab_file_save_path, 'phone39.txt')

    # Save mapping file
    if save_vocab_file:
        with open(phone61_vocab_map_file_path, 'w') as f:
            for phone in sorted(list(phone61_set)):
                f.write('%s\n' % phone)
        with open(phone48_vocab_map_file_path, 'w') as f:
            for phone in sorted(list(phone48_set)):
                f.write('%s\n' % phone)
        with open(phone39_vocab_map_file_path, 'w') as f:
            for phone in sorted(list(phone39_set)):
                f.write('%s\n' % phone)

    trans_dict = {}
    for label_path in tqdm(label_paths):
        speaker = label_path.split('/')[-2]
        utt_index = basename(label_path).split('.')[0]
        utt_name = speaker + '_' + utt_index

        phone61_list = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                # start_frame = line[0]
                # end_frame = line[1]
                phone61_list.append(line[2])

        # Map from 61 phones to the corresponding phones
        phone48_list = map_phone2phone(phone61_list, 'phone48',
                                       phone2phone_map_file_path)
        phone39_list = map_phone2phone(phone61_list, 'phone39',
                                       phone2phone_map_file_path)

        # Convert to string
        trans_phone61 = ' '.join(phone61_list)
        trans_phone48 = ' '.join(phone48_list)
        trans_phone39 = ' '.join(phone39_list)

        # for debug
        # print(trans_phone61)
        # print(trans_phone48)
        # print(trans_phone39)
        # print('-----')

        trans_dict[utt_name] = [trans_phone61, trans_phone48, trans_phone39]

    # Tokenize
    print('=====> Tokenize...')
    phone2idx_61 = Phone2idx(phone61_vocab_map_file_path)
    phone2idx_48 = Phone2idx(phone48_vocab_map_file_path)
    phone2idx_39 = Phone2idx(phone39_vocab_map_file_path)
    for utt_name, [trans_phone61, trans_phone48, trans_phone39] in tqdm(trans_dict.items()):
        if is_test:
            trans_dict[utt_name] = [
                trans_phone61, trans_phone48, trans_phone39]
            # NOTE: save as it is
        else:
            phone61_indices = phone2idx_61(trans_phone61)
            phone48_indices = phone2idx_48(trans_phone48)
            phone39_indices = phone2idx_39(trans_phone39)

            phone61_indices = ' '.join(
                list(map(str, phone61_indices.tolist())))
            phone48_indices = ' '.join(
                list(map(str, phone48_indices.tolist())))
            phone39_indices = ' '.join(
                list(map(str, phone39_indices.tolist())))

            trans_dict[utt_name] = [phone61_indices,
                                    phone48_indices, phone39_indices]
    return trans_dict
