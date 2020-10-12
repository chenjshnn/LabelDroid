# IMPLEMENTATION

## PREREQUISITE

Python 3.5
pytorch 1.1.0


## DATA

### IMAGE

<DATASET_FOLDER>
	|_ train
	  |_ **/**.png
	|_ val
	  |_ **/**.png
	|_test
	  |_ **/**.png

Image_name should be <imgid>.png

### ANNOTATION

<ANNOTATION_FOLDER>
	|_ annotations_train.json
	|_ annotations_val.json
	|_ annotations_test.json


Annotation:  Please refer to [COCO dataset format](http://cocodataset.org/#format-data)

*annotation_{split}.json*
{
 "annotations": [anno_item1, anno_item2, ...],
 "images": [img_item1, img_item2, img_item3, ...]
}

anno_item: {"id":int, "caption":str, "image_id":int}
img_item: {"id":int, "filename":"**/**.png", "height":int, weight:int}


## USAGE

### PREPARATION

* STEP 1: install all required packages *
``` pip install -r requirement ```

* STEP 2: prepare your data as above format *

* STEP 3: generate the vocab.pkl *

``` 
python data_utils/build_vocab.py --caption_path <PATH_TO_YOUR_TRAIN_ANNOTATION_JSON> --vocab_path <VOCAB_OUTPUT_PATH>.pkl
```

### TRAIN your own model
```
python train.py \
--image_root <PATH_TO_YOUR_DATASET_FOLDER> \
--caption_path <PATH_TO_ANNOTATION_FOLDER> \
--vocab_path <PATH_TO_GENERATED_VOCAB_FILE> \
--caption_model transformer \
--model_path run/models \
--num_epochs <NUM_EPOCH> \
--batch_size <BATCH_SIZE> 
```

*More options could be seen in opts.py*

### LOG
```
cd <PATH_TO_MODEL_DIR>
tensorboard --logdir=.
```


### TEST 
```
python3 test.py \
--image_root <PATH_TO_YOUR_DATASET_FOLDER> \
--caption_path <PATH_TO_ANNOTATION_FOLDER> \
--vocab_path <PATH_TO_GENERATED_VOCAB_FILE> \
--caption_model transformer \
--model_path <PATH_TO_TRAINED_MODEL_DIR> \
--batch_size <BATCH_SIZE> \
--split test \
```


Therefore, all results will saved to result.txt 

## ACKNOELEDGEMENTS
Some codes are modified from the following repositories. Thanks for their work.

https://github.com/aditya12agd5/convcap
https://github.com/yunjey/pytorch-tutorial
https://nlp.seas.harvard.edu/2018/04/03/attention.html