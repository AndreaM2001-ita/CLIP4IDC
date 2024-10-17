# CLIP4IDC: CLIP for Image Difference Captioning
## Requirement
```sh
# From CLIP
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
pip install "git+https://github.com/salaniz/pycocoevalcap.git"
pip install pycocoevalcap
```

## Data Preparing
**For Spot-the-Diff**
Either Restructure your dataset as spot the diff 

or Resized images can be download from [Learning to Describe Differences Between Pairs of Similar Images (EMNLP18)](https://github.com/harsh19/spot-the-diff/blob/master/data/get_images.txt). Raw captions can be download from [link](https://github.com/harsh19/spot-the-diff/tree/master/data/annotations). 

Collecting the captions belonging to the same image pair by running

```sh
python preprocess/reformat_dataset.py
```

For the convenience, you can also download the three json files from [link](https://drive.google.com/drive/folders/1g8QD6Y3La8cIamE7jeSSlXTw8G3r5Q8o?usp=sharing).

You would get

```
your_data_path
|–– clevr_change/
|   |–– data/
|   |   |–– images/
|   |   |–– nsc_images/
|   |   |–– sc_images/
|   |   |–– sc_images/
|   |   |–– change_captions.json
|   |   |–– no_change_captions.json
|   |   |–– splits.json
|   |   |–– type_mapping.json
|–– spot-the-diff/
|   |–– images/
|   |–– data/
|   |–– train.json
|   |–– val.json
|   |–– test.json
|   |–– reformat_train.json
|   |–– reformat_val.json
|   |–– reformat_test.json
```

## Prepare for Evaluation

**For Spot-the-Diff**

Running the command `python gt/eval_utils.py`, renaming the output file as `spot_total_change_captions_reformat.json`. You would get

```
gt
|–– clevr_total_change_captions_reformat.json
|–– spot_total_change_captions_reformat.json
```


# Pretrained Weight

```sh
cd ckpts
mkdir pretrained
mkdir trained
```

You can download the [Pretrained Weights](https://drive.google.com/drive/folders/1qOYVpZy57clJPF6AThsnO0Tfy4zq-gg1?usp=sharing) from the IDC Adaptation and the [Trained Weights](https://drive.google.com/drive/folders/18UfIvwKt0EE14EbogJycMmANpUJtsZbE?usp=sharing) from the IDC Finetuning. You would get

```
ckpts
|–– pretrained/
|   |–– pytorch_model.bin.clevr
|   |–– pytorch_model.bin.spot
|–– trained/
|   |–– pytorch_model.bin.clevr
|   |–– pytorch_model.bin.spot
```

The pretrained weights are the output of adaptation (retrieval) stage. The trained weights are the output of the finetuning (captioning) stage. 

## Adapation

Experiments are conducted on two NVIDIA **V100**. Time required for each task is less than 24h.

### Spot-the-Diff
```sh
DATA_PATH=images
python main_task_retrieval.py \
--do_train \
--num_thread_reader=4 \   
--epochs=20 \
--batch_size=128 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/images_test \
--output_dir ckpts/ckpt_spot_retrieval \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 128 \
--datatype spot \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32
```

## Finetuning

Time required for each task is less than 24h.

```sh
export JAVA_HOME="your_jdk/"
export PATH=$JAVA_HOME/bin:$PATH

DATA_PATH=images
python CLIP4IDCinference.py \
--do_train \
--num_thread_reader=4 \
--epochs=150 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/images_test \
--output_dir ckpts/ckpt_spot_caption \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 32 \
--datatype spot \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model pretrained/pytorch_model.bin.30
```

# Acknowledgments
Our code is largely borrowed from [CLIP](https://github.com/openai/CLIP), [UniVL](https://github.com/microsoft/UniVL) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).

# TODO
We are sorry that some lines of the code are redundant and some variables are named with "video". 
