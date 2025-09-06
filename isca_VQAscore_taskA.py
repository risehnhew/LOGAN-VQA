####

import pandas as pd
from scipy.stats import spearmanr, kendalltau
import torch
from t2v_metrics import VQAScore
import ast

import numpy as np
import os
from tqdm import tqdm
from utils import calculate_per_image_accuracy, ndcg, logan_vqa_scores

openai = False
MODE = 'logit_restriction' # get VQAscores for ranking # 'vqascore' or 'logit_restriction'
USE_ENSEMBLE = False          # True = average over all paraphrases
         # which template to use when not ensembling

print(MODE)
test_data = 'A-EN' # or 'A-PT'


# os.environ["HF_HOME"] = "/root/lanyun-tmp/models"

# model_names =  [ 'clip-flant5-xl', 'clip-flant5-xxl','llava-v1.5-13b', 'llava-v1.5-7b', 'sharegpt4v-7b', 'sharegpt4v-13b', 'llava-v1.6-13b'] #, 'clip-flant5-xxl','llava-v1.5-13b', 'llava-v1.5-7b', 'sharegpt4v-7b', 'sharegpt4v-13b', 'llava-v1.6-13b'
model_names = ['clip-flant5-xl'] #, 'clip-flant5-xxl','llava-v1.5-13b', 'llava-v1.5-7b', 'sharegpt4v-7b', 'sharegpt4v-13b', 'llava-v1.6-13b']
file_address = '/root/lanyun-tmp/stored-data/Subtask_A/EN/test/'
data_address = file_address +'subtask_a_test.tsv'
# file_names = ['Subtask_A/EN/xeval/subtask_a_xe.tsv','Subtask_A/PT/xeval/subtask_a_xp.tsv','Subtask_Bs/xeval/subtask_b_xe.tsv']
# file_dict = {
# 	'A-EN': '/lustre/projects/Research_Project-T127857/SemEval2025/revised_data2708/Subtask_A/EN/xeval/subtask_a_xe.tsv',
# 	'A-PT': 'Subtask_A/PT/xeval/subtask_a_xp.tsv',
# 	'B-EN': 'Subtask_B/xeval/subtask_b_xe.tsv'
# }

for model_name in model_names:
	# ['clip-flant5-xxl', 'clip-flant5-xl', 'clip-flant5-xxl-no-system', 'clip-flant5-xxl-no-system-no-user', 'llava-v1.5-13b', 'llava-v1.5-7b', 'sharegpt4v-7b', 'sharegpt4v-13b', 'llava-v1.6-13b', 'instructblip-flant5-xxl', 'instructblip-flant5-xl', 'gpt-4-turbo', 'gpt-4o']
	print(model_name)
	# Function to calculate per-image accuracy


	torch_dtype = torch.float16
	device = "cuda" if torch.cuda.is_available() else "cpu"
	

	if openai:
		score_func = t2v_metrics.get_score_model(model="gpt-4o", device="cuda", openai_key=openai_key, top_logprobs=20) # We find top_logprobs=20 to be sufficient for most (image, text) samples. Consider increase this number if you get errors (the API cost will not increase).
	else:
		score_func = VQAScore(model=model_name) # our recommended scoring model
	mwe_prompt_templates = [
		lambda c, cs: f"'The meaning of {c} in the sentence: {cs}'",
		lambda c, cs: f"An image representing '{c}' as used in: '{cs}'", # Your "Best"
		lambda c, cs: f"This image visually represents '{c}' as used in the sentence: '{cs}'",
		lambda c, cs: f"An illustration of '{c}' reflecting its specific meaning in the sentence: '{cs}'",
		lambda c, cs: f"A visual representation relevant to '{c}' given the context: '{cs}'",
		lambda c, cs: f"Image representing the concept of '{c}' based on its use in: '{cs}'"
	]	
	validation_num = len(mwe_prompt_templates)
	validation_scores = []
	for j in range(validation_num):

		cp_only = True
		train_data_info = pd.read_csv(data_address, sep='\t')
		compounds = train_data_info['compound'].tolist()
		context_sentences = train_data_info['sentence'].tolist()
		ground_t_orders = train_data_info['expected_order'].tolist()
		sentence_type = train_data_info['sentence_type'].tolist()
		# breakpoint()
		image_caption_dict = {}

		# Loop over the rows and fill the dictionary with the image names and their captions
		for _, row in train_data_info.iterrows():
			# Add image names and their corresponding captions to the dictionary
			for i in range(1,6):
				image_caption_dict[row['image{}_name'.format(i)]] = row['image{}_caption'.format(i)]
				# breakpoint()
		spearman_scores,ndcg_scores,kendall_taus,dcg_scores = [],[],[],[]
		correct_count = 0
		predict_rankings, ground_truth_rankings = [], []
		for compound, c_sent, g_t_order, sen_type in tqdm(zip(compounds, context_sentences, ground_t_orders,sentence_type)):
			address =  file_address+compound.replace("'", "_")+ '/'
			image_addresses = os.listdir(address)
			image_list = []
			for image_add in image_addresses:
				image_address = address+image_add
				
				image_list.append(image_address)
				single_cap = image_caption_dict[image_add]	
			
			sentences = [mwe_template_func(compound, c_sent)  for mwe_template_func in mwe_prompt_templates] #, new_sentence2, new_sentence3
			accumulated_scores = torch.zeros(len(image_list), device='cpu')
			# breakpoint()

			if USE_ENSEMBLE:
				# Optional: a neutral prior image (if you have one); otherwise leave as None
				neutral_img = None  # e.g., "/path/to/gray_224.png"

				# Optional: build foil prompts by swapping the sense term if you use them.
				# If not using foils, pass foil_texts=None.
				foil_prompts = None
				SINGLE_PROMPT_IDX = j
				texts_to_use = sentences if USE_ENSEMBLE else [sentences[SINGLE_PROMPT_IDX]]
				# Get LOGAN-VQA ensemble scores (higher is better)
				scores_t = logan_vqa_scores(
				    score_func,
				    images=image_list,
				    texts=texts_to_use,
				    prior_image=neutral_img,   # or None
				    foil_texts=foil_prompts,   # or None
				    aggregate="mean",  # or "harmonic" for conservative combining
				    mode=MODE          
				)

				# np.save("scores_logit_restriction.npy", scores_t.cpu().numpy())   # in the LOGIT path
			else: 



				# scores_t = score_func(images=image_list, texts=[sentences[j]], indicator=sen_type) # , question_template = "{}"get VQAscores for ranking # 'vqascore' or 'logit_restriction'
				
				scores_t = score_func(images=image_list, texts=[sentences[j]], mode=MODE) # , question_template = "{}"get VQAscores for ranking # 'vqascore' or 'logit_restriction'

				scores_t = scores_t.squeeze() # Make it (num_images,)

				# np.save("scores_baseline.npy", scores_t.cpu().numpy())
			# 	accumulated_scores += scores_t.cpu()
			# final_scores = accumulated_scores / len(sentences)

			# breakpoint()
			# forward(self,
			#         images: list[str],
			#         texts: list[str],
			#         question_template: str = default_question_template, # Make sure these defaults are accessible
			#         answer_template: str = default_answer_template,   # Used only in 'vqascore' mode
			#         mode: str = 'vqascore' # 'vqascore' or 'logit_restriction'
			#        ) -> torch.Tensor:
			# rank the images------
			# probs = temp_logit.flatten()

			predicted_indices = torch.argsort(scores_t, descending=True).cpu()
			predict_rankings.append(predicted_indices)
			# breakpoint()
			
			list_g_t = ast.literal_eval(g_t_order)
			ground_truth_indices = [image_addresses.index(image) for image in list_g_t]
			ground_truth_rankings.append(ground_truth_indices)

			# Calculate Spearman's rank correlation coefficient
			spearman_corr, _ = spearmanr(predicted_indices, ground_truth_indices)
			spearman_scores.append(spearman_corr)
			# breakpoint()
			predicted_indices_l = predicted_indices.tolist()
			# breakpoint()
			ndcg_score, dcg_score = ndcg(ground_truth_indices, predicted_indices_l)
			ndcg_scores.append(ndcg_score)
			dcg_scores.append(dcg_score)
			# breakpoint()
			kendall_taus.append(kendalltau(ground_truth_indices, predicted_indices_l)[0])

			# Top Image Accuray

			if ground_truth_indices[0] == predicted_indices[0]:

				correct_count += 1
		
		top_image_accuracy = correct_count / len(ground_t_orders)
		# print(f"Average Spearman's Rank Correlation Coefficient: {np.mean(spearman_scores)}")
		print(f"Top Image Accuracy: {top_image_accuracy:.2f}")
		print("NDCG:", round(sum(ndcg_scores) / len(ndcg_scores),3))
		print("DCG:", round(sum(dcg_scores) / len(dcg_scores), 3))

		# print("kendall_tau:", round(sum(kendall_taus)/len(kendall_taus),3))

		# accuracy_all = calculate_per_image_accuracy(ground_truth_rankings, predict_rankings)
		# print(f"Over all accuracy:{round(accuracy_all, 3)}")
		# validation_scores.append(top_image_accuracy)

				# Convert lists of rankings to DataFrame-friendly format
		# (store as strings so each row is easy to read)
		# df_out = pd.DataFrame({
		#     "compound": compounds,
		#     "context_sentence": context_sentences,
		#     "ground_truth_order": [str(x) for x in ground_truth_rankings],
		#     "predicted_order": [str(x.tolist() if hasattr(x, "tolist") else x) for x in predict_rankings]
		# })

		# # Save to CSV
		# output_file = f"predicted_orders_{test_data}_{model_name}_{j}.csv"
		# df_out.to_csv(output_file, index=False, encoding="utf-8")

		# print(f"Predicted orders saved to {output_file}")


	# final_weights = np.array(validation_scores) / sum(validation_scores)
	# scores = np.array(validation_scores)



	# # 1. Normalized Scores
	# normalized = scores / np.sum(scores)

	# # 2. Softmax (default temperature)
	# softmax = np.exp(scores) / np.sum(np.exp(scores))

	# # 3. Softmax with temperature = 0.5 (sharper)
	# temp = 0.5
	# softmax_temp = np.exp(scores / temp) / np.sum(np.exp(scores / temp))

	# # 4. Rank-based (1/rank)
	# ranks = scores.argsort()[::-1].argsort() + 1  # Rank 1 = highest score
	# rank_weights = 1 / ranks
	# rank_weights /= np.sum(rank_weights)

	# # 5. Quadratic
	# quadratic = scores**2
	# quadratic /= np.sum(quadratic)

	# # 6. Exponential
	# exponential = np.exp(scores)
	# exponential /= np.sum(exponential)

	# # Combine into a DataFrame for easy comparison
	# df = pd.DataFrame({
	#     "Score": scores,
	#     "Normalized": normalized,
	#     "Softmax": softmax,
	#     "Softmax (T=0.5)": softmax_temp,
	#     "Rank-based": rank_weights,
	#     "Quadratic": quadratic,
	#     "Exponential": exponential
	# })

	# print(f"validation weights:")
	# print(rank_weights)