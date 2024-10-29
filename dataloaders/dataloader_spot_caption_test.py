from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import random
from dataloaders.rawimage_util import RawImageExtractor
from collections import defaultdict


class SPOT_DataLoader(Dataset):
    """SPOT dataset loader."""
    def __init__(
          self,
          subset,
          data_path,
          features_path,
          tokenizer,
          max_words=30,
          image_resolution=224,
          ):
      self.data_path = data_path
      self.features_path = features_path
      self.max_words = max_words
      self.tokenizer = tokenizer

      self.subset = subset
      assert self.subset in ["train", "val", "test"]

      if self.subset == "test":
        # Count images in the image_features folder
        image_files = [
            f for f in os.listdir(self.features_path)
            if os.path.isfile(os.path.join(self.features_path, f))
        ]
        self.image_num = len(image_files)//2

        # Logging for verification
        print("Subset: test")
        print("Total number of images", self.image_num)

        # Initialize empty structures as captions are not needed
        self.sentences_dict = {}
        self.cut_off_points = []
        self.sample_len = self.image_num  # Set sample length to image count
        self.image_dict = {}
        count = 0  # Initialize a counter for valid images

        # Loop through image files and add valid images to the dictionary
        for f in image_files:
            if not f.endswith("_2.png"):  # Exclude images ending with _2
                self.image_dict[count] = os.path.splitext(f)[0]  # Add to image_dict with count as key
                count += 1
  
        #assert len(self.cut_off_points) == self.image_num

        # Logging for verification
        print("Total number of valid images:", count)

      else:
          # Load captions from JSON file if not testing
          change_caption_file = os.path.join(self.data_path, f"reformat_{self.subset}.json")
          with open(change_caption_file, 'r') as fp:
              change_captions = json.load(fp)

          self.sample_len = 0
          self.sentences_dict = {}
          self.cut_off_points = []

          for cap in change_captions:
              image_id = cap["img_id"]
              self.sentences_dict[len(self.sentences_dict)] = (image_id, cap["sentences"])
              self.cut_off_points.append(len(self.sentences_dict))

          # Setup for multi-sentence retrieval for val and test
          self.multi_sentence_per_pair = True
          if self.subset == "val":
              self.sentence_num = len(self.sentences_dict)
              self.image_num = len(change_captions)
              assert len(self.cut_off_points) == self.image_num
              print(f"For {self.subset}, sentence number:", self.sentence_num)
              print(f"For {self.subset}, image number:", self.image_num)

          print("Image number:", len(change_captions))
          print("Total Pairs:", len(self.sentences_dict))

          self.sample_len = len(self.sentences_dict)

      # Initialize image extractor and special tokens
      self.rawImageExtractor = RawImageExtractor(size=image_resolution)
      self.SPECIAL_TOKEN = {
          "CLS_TOKEN": "<|startoftext|>",
          "SEP_TOKEN": "<|endoftext|>",
          "MASK_TOKEN": "[MASK]",
          "UNK_TOKEN": "[UNK]",
          "PAD_TOKEN": "[PAD]"
      }


    def __len__(self):
        return self.sample_len

    def _get_text(self, image_id, caption):
        k = 1
        choice_image_ids = [image_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.int64)

        for i, image_id in enumerate(choice_image_ids):
            words = []

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + caption_words
            output_caption_words = caption_words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids

    def _get_rawimage(self, image_path):
        choice_image_path = [image_path]
        # Pair x L x T x 3 x H x W
        image = np.zeros((len(choice_image_path), 3, self.rawImageExtractor.size,
                          self.rawImageExtractor.size), dtype=float)

        for i, image_path in enumerate(choice_image_path):

            raw_image_data = self.rawImageExtractor.get_image_data(image_path)
            raw_image_data = raw_image_data['image']

            image[i] = raw_image_data

        return image

    def __getitem__(self, idx):
      if self.subset == "test":
        image_id=self.image_dict[idx]
        
        bef_image_path = os.path.join(self.features_path, "%s.png" % image_id)
        aft_image_path = os.path.join(self.features_path, "%s_2.png" % image_id)
        image_idx_name = "%s.png" % image_id

        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        image_mask = np.ones(2, dtype=np.int64)
        
        k = 1
        choice_image_ids = [image_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.int64)

        return pairs_text, pairs_mask, pairs_segment, bef_image, aft_image, image_mask, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, image_idx_name
      else:
        image_id, caption = self.sentences_dict[idx]
        caption = random.choice(caption)
        bef_image_path = os.path.join(self.features_path, "%s.png" % image_id)
        aft_image_path = os.path.join(self.features_path, "%s_2.png" % image_id)
        image_idx_name = "%s.png" % image_id

        pairs_text, pairs_mask, pairs_segment, pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = self._get_text(image_id, caption)
        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        image_mask = np.ones(2, dtype=np.int64)
        return pairs_text, pairs_mask, pairs_segment, bef_image, aft_image, image_mask, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, image_idx_name


