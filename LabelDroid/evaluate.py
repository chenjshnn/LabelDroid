"""
From Karpathy's neuraltalk2:
https://github.com/karpathy/neuraltalk2
Specifically:
https://github.com/karpathy/neuraltalk2/blob/master/coco-caption/myeval.py
"""

import sys
#sys.path.insert(0, '/media/cheer/UI/COCO2014')

import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys


def language_eval(args, input_data, savedir, split):
  if type(input_data) == str: # Filename given.
    checkpoint = json.load(open(input_data, 'r'))
    preds = checkpoint
  elif type(input_data) == list: # Direct predictions give.
    preds = input_data

  annFile = os.path.join(args.caption_path,'captions_{}.json'.format(split))
  coco = COCO(annFile)
  valids = coco.getImgIds()

  # Filter results to only those in MSCOCO validation set (will be about a third)
  preds_filt = [p for p in preds if p['image_id'] in valids]
  preds_filt = [{'image_id':p['image_id'], 'caption':p['caption']} for p in preds_filt]
  #print(preds_filt)
  print(('Using %d/%d predictions' % (len(preds_filt), len(preds))))
  resFile = osp.join(savedir, 'result_%s.json' % (split))
  json.dump(preds_filt, open(resFile, 'w')) # Serialize to temporary json file. Sigh, COCO API...

  cocoRes = coco.loadRes(resFile)
  cocoEval = COCOEvalCap(coco, cocoRes)
  cocoEval.params['image_id'] = cocoRes.getImgIds()
  cocoEval.evaluate()

  # Create output dictionary.
  out = {}
  for metric, score in list(cocoEval.eval.items()):
    out[metric] = score

  exact_match_num = 0
  for p in preds:
    if p["caption"]== p["gt_caption"]:
      exact_match_num += 1
  out["Exact_match"] = exact_match_num/len(preds)

  print("Exact_match:", out["Exact_match"])
  # Return aggregate and per image score.
  return out, cocoEval.evalImgs
