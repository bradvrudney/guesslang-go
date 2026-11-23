"""Run once, inside Docker, with tensorflow==2.5.0 and guesslang==2.2.1.

Writes, next to this file's parent:
  model/weights.json   the variables of guesslang's model, as nested lists
  model/classes.json   the language names, in the order of the model's outputs
  testdata/reference.json   the original's probabilities for every sample
"""
import json, os, sys, glob
import numpy as np
import tensorflow as tf
from guesslang import Guess
import guesslang

root = sys.argv[1]
mdir = os.path.join(os.path.dirname(guesslang.__file__), 'data', 'model')
reader = tf.train.load_checkpoint(os.path.join(mdir, 'variables', 'variables'))
names = sorted(reader.get_variable_to_shape_map())
print('variables:')
for n in names:
    print(' ', n, reader.get_variable_to_shape_map()[n])
weights = {n: reader.get_tensor(n).astype('float32').tolist() for n in names if 'Adagrad' not in n and 'global_step' not in n}
os.makedirs(os.path.join(root, 'model'), exist_ok=True)
json.dump(weights, open(os.path.join(root, 'model', 'weights.json'), 'w'))

g = Guess()
ref = {}
for f in sorted(glob.glob(os.path.join(root, 'testdata', 'samples', '*.txt'))):
    text = open(f, encoding='utf-8').read()
    probs = g.probabilities(text)
    ref[os.path.basename(f)] = {
        'probabilities': [[n, p] for n, p in probs],
        'language_name': g.language_name(text),
    }
json.dump(ref, open(os.path.join(root, 'testdata', 'reference.json'), 'w'), indent=1)
print('reference for', len(ref), 'samples')
