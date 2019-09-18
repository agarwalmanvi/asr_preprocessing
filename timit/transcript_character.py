#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Make character-level target labels for the End-to-End model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import re
from tqdm import tqdm

import numpy as np


class Char2idx(object):
    """Convert from character to index.
    Args:
        vocab_file_path (string): path to the vocabulary file
        space_mark (string, optional): the space mark to divide a sequence into words
        capital_divide (bool, optional): if True, words will be divided by
            capital letters. This is used for English.
        double_letter (bool, optional): if True, group repeated letters.
            This is used for Japanese.
        remove_list (list, optional): characters to neglect
    """

    def __init__(self, vocab_file_path, space_mark='_', capital_divide=False,
                 double_letter=False, remove_list=[]):
        self.space_mark = space_mark
        self.capital_divide = capital_divide
        self.double_letter = double_letter
        self.remove_list = remove_list

        # Read the vocabulary file
        self.map_dict = {}
        vocab_count = 0
        with open(vocab_file_path, 'r') as f:
            for line in f:
                char = line.strip()
                if char in remove_list:
                    continue
                self.map_dict[char] = vocab_count
                vocab_count += 1

        # Add <SOS> & <EOS>
        self.map_dict['<'] = vocab_count
        self.map_dict['>'] = vocab_count + 1

    def __call__(self, str_char):
        """
        Args:
            str_char (string): a sequence of characters
        Returns:
            index_list (list): character indices
        """
        index_list = []

        # Convert from character to index
        if self.capital_divide:
            for word in str_char.split(self.space_mark):
                # Replace the first character with the capital letter
                index_list.append(self.map_dict[word[0].upper()])

                # Check double-letters
                skip_flag = False
                for i in range(1, len(word) - 1, 1):
                    if skip_flag:
                        skip_flag = False
                        continue

                    if not skip_flag and word[i:i + 2] in self.map_dict.keys():
                        index_list.append(self.map_dict[word[i:i + 2]])
                        skip_flag = True
                    else:
                        index_list.append(self.map_dict[word[i]])

                # Final character
                if not skip_flag:
                    index_list.append(self.map_dict[word[-1]])

        elif self.double_letter:
            skip_flag = False
            for i in range(len(str_char) - 1):
                if skip_flag:
                    skip_flag = False
                    continue

                if not skip_flag and str_char[i:i + 2] in self.map_dict.keys():
                    index_list.append(self.map_dict[str_char[i:i + 2]])
                    skip_flag = True
                else:
                    index_list.append(self.map_dict[str_char[i]])

            # Final character
            if not skip_flag:
                index_list.append(self.map_dict[str_char[-1]])

        else:
            index_list = list(map(lambda x: self.map_dict[x], list(str_char)))

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

# NOTE:
############################################################
# [character]
# 26 alphabets(a-z)
# space(_), apostorophe(')
# = 30 labels

# [character_capital_divide]
# 26 lower alphabets(a-z), 26 upper alphabets(A-Z),
# 19 special double-letters, apostorophe(')
# = 74 labels
############################################################

DOUBLE_LETTERS = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj',
                  'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'ss', 'tt',
                  'uu', 'vv', 'ww', 'xx', 'yy', 'zz']
SPACE = '_'
APOSTROPHE = '\''


def read_char(label_paths, vocab_file_save_path, save_vocab_file=False,
              is_test=False):
    """Read text transcript.
    Args:
        label_paths (list): list of paths to label files
        vocab_file_save_path (string): path to vocabulary files
        save_vocab_file (string): if True, save vocabulary files
        is_test (bool, optional): set True in case of the test set
    Returns:
        trans_dict (dict):
            key (string) => utterance name
            value (list) => [char_indices, char_indices_capital]
    """
    print('=====> Reading target labels...')
    trans_dict = {}
    char_set, char_capital_set = set([]), set([])
    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as f:
            line = f.readlines()[-1]
            speaker = label_path.split('/')[-2]
            utt_index = basename(label_path).split('.')[0]
            utt_name = speaker + '_' + utt_index

            # Remove 「"」, 「:」, 「;」, 「！」, 「?」, 「,」, 「.」, 「-」
            # Convert to lowercase
            line = re.sub(r'[\":;!?,.-]+', '', line.strip().lower())

            transcript = ' '.join(line.split(' ')[2:])

            # Remove double spaces
            while '  ' in transcript:
                transcript = re.sub(r'  ', ' ', transcript)

            # Remove first and last space
            if transcript[0] == ' ':
                transcript = transcript[1:]
            if transcript[-1] == ' ':
                transcript = transcript[:-1]

            # Capital-divided
            for word in transcript.split(' '):
                if len(word) == 1:
                    char_capital_set.add(word.upper())
                else:
                    # Replace the first character with the capital letter
                    word = word[0].upper() + word[1:]
                    char_capital_set.add(word[0].upper())

                    # Check double-letters
                    skip_flag = False
                    for i in range(1, len(word) - 1, 1):
                        if skip_flag:
                            skip_flag = False
                            continue

                        if not skip_flag and word[i:i + 2] in DOUBLE_LETTERS:
                            char_capital_set.add(word[i:i + 2])
                            skip_flag = True
                        else:
                            char_capital_set.add(word[i])

                    # Final character
                    if not skip_flag:
                        char_capital_set.add(word[-1])

            # Convert space to "_"
            transcript = re.sub(r'\s', SPACE, transcript)

            for c in list(transcript):
                char_set.add(c)

            trans_dict[utt_name] = transcript

            # for debug
            # print(transcript)
            # print(trans_char_capital_divide)

    # Make vocabulary files
    char_vocab_file_path = mkdir_join(vocab_file_save_path, 'character.txt')
    char_capital_vocab_file_path = mkdir_join(
        vocab_file_save_path, 'character_capital_divide.txt')

    # Reserve some indices
    char_set.discard(SPACE)
    char_set.discard(APOSTROPHE)
    char_capital_set.discard(APOSTROPHE)

    # for debug
    # print(sorted(list(char_set)))
    # print(sorted(list(char_capital_set)))

    if save_vocab_file:
        # character-level
        with open(char_vocab_file_path, 'w') as f:
            char_list = sorted(list(char_set)) + [SPACE, APOSTROPHE]
            for char in char_list:
                f.write('%s\n' % char)

        # character-level (capital-divided)
        with open(char_capital_vocab_file_path, 'w') as f:
            char_capital_list = sorted(list(char_capital_set)) + [APOSTROPHE]
            for char in char_capital_list:
                f.write('%s\n' % char)

    # Tokenize
    print('=====> Tokenize...')
    char2idx = Char2idx(char_vocab_file_path)
    char2idx_capital = Char2idx(
        char_capital_vocab_file_path, capital_divide=True)
    for utt_name, transcript in tqdm(trans_dict.items()):
        if is_test:
            trans_dict[utt_name] = [transcript, transcript]
            # NOTE: save as it is
        else:
            char_indices = char2idx(transcript)
            char_indices_capital = char2idx_capital(transcript)

            char_indices = ' '.join(list(map(str, char_indices.tolist())))
            char_indices_capital = ' '.join(
                list(map(str, char_indices_capital.tolist())))

            trans_dict[utt_name] = [char_indices, char_indices_capital]

    return trans_dict
