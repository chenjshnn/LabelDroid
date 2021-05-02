# IMPLEMENTATION

## Prerequisite

- Python 3.5

- pytorch 1.1.0


## Data Structure

### Image

```
<DATASET_FOLDER>
├── train
   ├── **/**.png
├── val
   ├── **/**.png
├──test
   ├── **/**.png
```

Image_name should be <imgid>.png

### Annotation

```
<ANNOTATION_FOLDER>
├── annotations_train.json
├── annotations_val.json
├── annotations_test.json
```

Annotation:  Please refer to [COCO dataset format](http://cocodataset.org/#format-data)

*annotation_{split}.json*
```
{
 "annotations": [anno_item1, anno_item2, ...],
 "images": [img_item1, img_item2, img_item3, ...]
}

anno_item = {"id":int, "caption":str, "image_id":int}

img_item = {"id":int, "filename":"**/**.png", "height":int, weight:int}
```

## Usage

### Preparation

* STEP 1: install all required packages 

``` pip install -r requirement ```

* STEP 2: prepare your data as above format

* STEP 3: generate the vocab.pkl 

``` 
python data_utils/build_vocab.py \
--caption_path <PATH_TO_YOUR_TRAIN_ANNOTATION_JSON> \
--vocab_path <VOCAB_OUTPUT_PATH>.pkl
```

### Training
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

### Logging
```
cd <PATH_TO_MODEL_DIR>
tensorboard --logdir=.
```


### Testing 


# this code finds the testing images via <i>a json file</i> (same format as the coco dataset)

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

See [test_transformer.sh](test_transformer.sh) as an example

All results will be saved to result.json in the trained model folder 

# this code finds the testing images via <i>an image folder</i> which stores the images you want to test
```
python3 sample.py \
--image_root sample \
--vocab_path ./data/vocab.pkl \
--caption_model transformer \
--model_path run/models_transformer/best_model.ckpt \
--batch_size 4 
```

See [sample.sh](sample.sh) as an example

All results will be saved to result.json in the image folder 




## Acknowledgment
Some codes are based on the following repositories. Thanks for their work.

- https://github.com/aditya12agd5/convcap

- https://github.com/yunjey/pytorch-tutorial

- https://nlp.seas.harvard.edu/2018/04/03/attention.html