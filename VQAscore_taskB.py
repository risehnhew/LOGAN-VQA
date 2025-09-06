####

import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from t2v_metrics import t2v_metrics
import ast

import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from utils import evaluate_completion_accuracy, evaluate_labeling_f1_score, choose_prompts


# model_name = 'clip-flant5-xl' #'llava-v1.5-13b'
model_names = ['llava-v1.5-13b','llava-v1.6-13b']
# model_names = ['gpt-4o']

data_types = ['train', 'dev', 'test']
# Function to calculate per-image accuracy



torch_dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"
# breakpoint()
for data_t in data_types:
	print(data_t)
	for model_name in model_names:
		print(model_name)

		if model_name=='gpt-4o':
			score_func = t2v_metrics.get_score_model(model=model_name, device="cuda", openai_key=openai_key, top_logprobs=20) # We find top_logprobs=20 to be sufficient for most (image, text) samples. Consider increase this number if you get errors (the API cost will not increase).
		else:
			score_func = t2v_metrics.VQAScore(model=model_name) # our recommended scoring model


		cp_only = True
		train_data_info = pd.read_csv('/root/lanyun-tmp/stored-data2/upload/B-en/xeval/subtask_a_test.tsv', sep='\t')
		compounds = train_data_info['compound'].tolist()
		# context_sentences = train_data_info['Sentence'].tolist()
		expected_items = train_data_info['expected_item'].tolist()
		sentence_type = train_data_info['sentence_type'].tolist()
		# meanings = train_data_info['Meaning'].tolist()
		image_caption_dict = {}

		# Loop over the rows and fill the dictionary with the image names and their captions
		for _, row in train_data_info.iterrows():
			# Add image names and their corresponding captions to the dictionary
			for i in range(1,5):
				image_caption_dict[row['image{}_name'.format(i)]] = row['image{}_caption'.format(i)]
			for j in range(1,3):
				image_caption_dict[row['compound']+'_s{}.png'.format(j)] = row['sequence_caption{}'.format(j)]
				# breakpoint()
		spearman_scores = []
		correct_count = 0
		predict_rankings, ground_truth_rankings = [], []
		predicted_labels =[]

		New_sentences = []

		for compound, expected_item, sen_type in tqdm(zip(compounds,expected_items,sentence_type)):

			address = f'/root/lanyun-tmp/stored-data2/upload/B-en/xeval/{compound}/'
			# combined_folder = address+'combined3/'
			combined_folder = address
			
			image_addresses =  [
		    f for f in os.listdir(combined_folder)
		    if os.path.isfile(os.path.join(combined_folder, f)) and f not in {'s1.png', 's2.png'}
			] #files only

			# breakpoint()
			image_list = []
			logits = []
			first_two_image = [address+'s1.png', address+'s2.png']

			first_two_address = []
			
			# breakpoint()
			if not os.path.exists(combined_folder):
				os.mkdir(combined_folder)
			for image_add in image_addresses:
				combined_image = combined_folder+ image_add	
				image_list.append(combined_image)		
			# breakpoint()
				
			# new_sentence = 'The ' +sen_type+ ' meaning of '+ compound #
			# new_sentence = 'The meaning of '+ compound+ ' in the sentence: ' + c_sent #
			# new_sentence = meaning
			new_sentence = choose_prompts(sen_type, compound)
			New_sentences.append(new_sentence)
				
				# new_sentence = sen_type + compound
				# new_sentence =  compound


				# logits.append(logit)
			# breakpoint()
			logits_t = score_func(images=image_list, texts=[new_sentence])

			paired_list = list(zip(logits_t.flatten().tolist(), image_list))
			# Sorting paired list by logits_t in descending order
			sorted_paired_list = sorted(paired_list, key=lambda x: x[0], reverse=True)
			# Extracting the sorted image_list
			sorted_image_list = [item[1] for item in sorted_paired_list]
			print(new_sentence)
			predicted_labels.append(sorted_image_list[0].replace(combined_folder,''))  

		# breakpoint()
		# print(New_sentences)
		print(combined_folder)
		evaluate_completion_accuracy(expected_items, predicted_labels, compounds, sentence_type)
		evaluate_labeling_f1_score(expected_items, predicted_labels)

