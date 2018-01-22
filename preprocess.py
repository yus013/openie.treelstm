"""
Preprocessing script for SICK data.

"""

import os
import glob
import json


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)


def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
        % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)


def build_vocab(filepaths, dst_path):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        json.dump(list(vocab), fp=f)


def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'sent.txt'), cp=cp, tokenize=True)
    constituency_parse(os.path.join(dirpath, 'sent.txt'), cp=cp, tokenize=True)


# TODO: wait for real dataset
if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    lib_dir = os.path.join(base_dir, 'lib')

    sent_dir = os.path.join(data_dir, 'sent')  # raw sentence
    train_sent_dir = os.path.join(sent_dir, 'train')
    # dev_sent_dir = os.path.join(sent_dir, 'dev')
    # test_sent_dir = os.path.join(sent_dir, 'test')
    # make_dirs([train_sent_dir, dev_sent_dir, test_sent_dir])
    make_dirs([train_sent_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    # parse sentences
    parse(train_sent_dir, cp=classpath)
    # parse(dev_sent_dir, cp=classpath)
    # parse(test_sent_dir, cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(sent_dir, '*/*.toks')),  # glob: find files matching pattern
        os.path.join(data_dir, 'vocab.txt'))

    # extract a-r-b
    # TODO: wait for help
    arb_dir = os.path.join(data_dir, 'arb')
    train_arb_dir = os.path.join(arb_dir, 'train')  # arb
    # dev_arb_dir = os.path.join(arb_dir, 'dev')
    # test_arb_dir = os.path.join(arb_dir, 'test')
    # make_dirs([train_arb_dir, dev_arb_dir, test_arb_dir])
    make_dirs([train_arb_dir])
