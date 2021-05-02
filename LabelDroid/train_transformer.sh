python3 train.py \
--image_root Accessibility_Dataset \
--caption_path annotation \
--vocab_path ./data/vocab.pkl \
--caption_model transformer \
--model_path run/models \
--num_epochs 500 \
--batch_size 64  

