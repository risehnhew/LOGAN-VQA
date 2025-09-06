from sklearn.metrics import accuracy_score, f1_score
import math


def evaluate_completion_accuracy(true_labels, predicted_labels, noun_compounds, sen_types):
    """
    Evaluates the Completion Accuracy by comparing true labels with predicted labels,
    and identifies correctly predicted noun compounds and their sentence types.
    
    Args:
    true_labels (list): A list of ground truth labels for the sequence completion task.
    predicted_labels (list): A list of labels predicted by the model.
    noun_compounds (list): A list of noun compounds corresponding to the labels.
    sen_types (list): A list of sentence types corresponding to the labels.
    
    Returns:
    tuple: A tuple containing the accuracy score, a list of correctly predicted noun compounds,
           and their corresponding sentence types.
    """
    # Ensure the inputs have the same length
    assert len(true_labels) == len(predicted_labels) == len(noun_compounds) == len(sen_types), \
        "All input lists must have the same length."
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Identify correctly predicted noun compounds and their sentence types
    correctly_predicted = [
        (compound, sen_type) 
        for label, pred, compound, sen_type in zip(true_labels, predicted_labels, noun_compounds, sen_types) 
        if label == pred
    ]
    
    # Extract compounds and sentence types separately for easier interpretation
    correct_noun_compounds = [item[0] for item in correctly_predicted]
    correct_sen_types = [item[1] for item in correctly_predicted]
    
    print(f"Completion Accuracy: {accuracy}")
    print(f"Correctly Predicted Noun Compounds: {correct_noun_compounds}")
    print(f"Correctly Predicted Sentence Types: {correct_sen_types}")
    
    return accuracy, correct_noun_compounds, correct_sen_types
def evaluate_labeling_f1_score(true_labels, predicted_labels):
    """
    Evaluates the F1 Score for distinguishing between idiomatic and literal expressions.
    
    Args:
    true_labels (list): A list of ground truth labels indicating idiomatic or literal meaning.
    predicted_labels (list): A list of labels predicted by the model.
    
    Returns:
    float: The F1 score of the model's predictions.
    """
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    print(f"Labeling F1 Score: {f1:.2f}")
    return f1
def choose_prompts(sen_type, noun_compound):

    if sen_type == 'idiomatic':
        prompt = 'The idiomatic mean of \'' + noun_compound +'\'.'
    elif sen_type == 'literal':
        # prompt = 'If we interpret \''+ noun_compound + '\' word by word, its literal mean.'
        prompt = 'If we interpret \''+ noun_compound + '\' word by word, what would it mean literally?'
    else:
        print('Prompt ERROR')
    return prompt
def calculate_per_image_accuracy(ground_truth_rankings, predicted_rankings):
    total_images = 0
    correct_images = 0

    # Loop over each pair of ground truth and predicted rankings
    for ground_truth, prediction in zip(ground_truth_rankings, predicted_rankings):
        # Loop over each image in the ground truth and predicted rankings
        for gt_image, pred_image in zip(ground_truth, prediction):
            total_images += 1
            # Check if the image is in the correct position
            if gt_image == pred_image:
                correct_images += 1

    # Calculate the per-image accuracy
    per_image_accuracy = correct_images / total_images
    return per_image_accuracy
# def dcg(relevance_scores):
#     """Compute Discounted Cumulative Gain (DCG) given a list of relevance scores."""
#     return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

# def ndcg(ground_truth_indices, predicted_indices):
#     """
#     Compute NDCG given the ground truth order and the predicted order.

#     Parameters:
#     - ground_truth_indices: list of item indices in ideal order (best to worst).
#     - predicted_indices: list of item indices in the order predicted by the model.

#     Returns:
#     - ndcg_score: The NDCG score (float).
#     """
#     n = len(ground_truth_indices)
#     # Assign relevance scores based on ground truth position:
#     # The top item gets a relevance of n, the next gets n-1, etc.
#     relevance_map = {item: n - i for i, item in enumerate(ground_truth_indices)}
#     # breakpoint()
#     # Get the relevance scores in the predicted order
#     predicted_relevance = [relevance_map[item] for item in predicted_indices]

#     # The ideal relevance scores are simply the relevance scores in the ground truth order
#     ideal_relevance = [relevance_map[item] for item in ground_truth_indices]

#     # Compute DCG (predicted) and IDCG (ideal)
#     dcg_val = dcg(predicted_relevance)
#     idcg_val = dcg(ideal_relevance)

#     # Return NDCG
#     return dcg_val / idcg_val if idcg_val > 0 else 0.0, dcg_val
# import math

def dcg(relevance_scores, *, exp_gain=False):
    """
    relevance_scores: list[float|int] aligned to the predicted ranking (top -> bottom)
    exp_gain: if True, uses 2^rel - 1 (exponential gain); else linear gain
    """
    dcg_val = 0.0
    for idx, rel in enumerate(relevance_scores):   # idx is 0-based rank
        gain = (2.0 ** rel - 1.0) if exp_gain else float(rel)
        dcg_val += gain / math.log2(idx + 2.0)    # log2(rank+1), rank starts at 1
    return dcg_val

def ndcg(ground_truth_indices, predicted_indices, *, k=None, exp_gain=False):
    """
    ground_truth_indices: ideal order (best -> worst)
    predicted_indices: model order (best -> worst)
    k: optional cutoff (e.g., 5)
    exp_gain: use exponential gains if True
    Returns: (ndcg_score, dcg_val, idcg_val)
    """
    # Make sure inputs are lists of hashables
    gt = list(ground_truth_indices)
    pred = list(predicted_indices)

    n = len(gt)
    if k is None:
        k = n
    k = min(k, len(pred), n)

    # Graded relevance: top gets n, next n-1, ..., last gets 1
    relevance_map = {item: (n - i) for i, item in enumerate(gt)}

    # Relevance in predicted order (items not in GT get 0)
    predicted_relevance = [relevance_map.get(item, 0) for item in pred[:k]]

    # Ideal relevance at the same cutoff
    ideal_relevance = [relevance_map[item] for item in gt[:k]]

    dcg_val  = dcg(predicted_relevance, exp_gain=exp_gain)
    idcg_val = dcg(ideal_relevance,    exp_gain=exp_gain)
    ndcg_val = (dcg_val / idcg_val) if idcg_val > 0 else 0.0
    return ndcg_val, dcg_val



def ndcg_leaderboard(ground_truth_order, predicted_order, k=None):
    """
    Compute DCG and NDCG using leaderboard-style definition:
    - binary relevance: 1 if predicted item at position i == ground_truth item at position i, else 0
    - log base 2 discount
    - cutoff @k (default: full length)

    Returns:
        ndcg (float), dcg (float), idcg (float)
    """
    n = min(len(ground_truth_order), len(predicted_order))
    if k is None:
        k = n
    else:
        k = min(k, n)

    # DCG: binary per-position correctness
    dcg = 0.0
    for i in range(k):
        rel = 1 if predicted_order[i] == ground_truth_order[i] else 0
        dcg += rel / math.log2(i + 2)

    # IDCG: ideal = all correct at each position (all rel=1)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(k))

    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return ndcg, dcg

# Example usage:
# Suppose you have 5 items (0,1,2,3,4).
# predicted_scores[i] is the predicted score for item i.
# predicted_scores = [0.8, 0.3, 0.9, 0.1, 0.5]  # Example predicted scores

# ground_truth_order is a ranking of items from best to worst (e.g., item 2 is best, then 0, etc.)
# ground_truth_order = [2, 0, 4, 1, 3]

# ndcg_value = ndcg(predicted_scores, ground_truth_order)
# print("NDCG:", ndcg_value)




# If I use the meaning got from GPT4o:
# instructblip-flant5-xl
# Completion Accuracy: 0.55
# Correctly Predicted Noun Compounds: ['cold turkey', 'best man', 'new blood', 'open goal', 'guinea pig', 'dead wood', 'panda car', 'bells and whistles', 'silver bullet', 'cutting edge', 'dutch courage']
# Correctly Predicted Sentence Types: ['idiomatic', 'idiomatic', 'idiomatic', 'idiomatic', 'literal', 'idiomatic', 'idiomatic', 'idiomatic', 'idiomatic', 'idiomatic', 'literal']
# Labeling F1 Score: 0.55




# If I use enhanced prompts

# instructblip-flant5-xl

# Completion Accuracy: 0.45
# Correctly Predicted Noun Compounds: ['cold turkey', 'seal of approval', 'best man', 'agony aunt', 'guinea pig', 'bells and whistles', 'swan song', 'dutch courage', 'sour grapes']
# Correctly Predicted Sentence Types: ['idiomatic', 'literal', 'idiomatic', 'literal', 'literal', 'idiomatic', 'idiomatic', 'literal', 'literal']
# Labeling F1 Score: 0.45


# OpenAI GPT4o:

# ./AdMIRe_Subtask_B_Train/train/sour grapes/
# Completion Accuracy: 0.45
# Correctly Predicted Noun Compounds: ['seal of approval', 'best man', 'new blood', 'white whale', 'dead wood', 'bells and whistles', 'swan song', 'cutting edge', 'dutch courage']
# Correctly Predicted Sentence Types: ['literal', 'idiomatic', 'idiomatic', 'literal', 'idiomatic', 'idiomatic', 'idiomatic', 'idiomatic', 'literal']
# Labeling F1 Score: 0.45
# utils.py
import torch
from typing import List, Optional

@torch.no_grad()
def logan_vqa_scores(
    score_func,
    images: List[str],
    texts: List[str],
    *,
    prior_image: Optional[str] = None,           # optional neutral image
    foil_texts: Optional[List[str]] = None,      # optional negated/contrast prompts
    aggregate: str = "mean",
    mode="vqascore"                     # "mean" or "harmonic"
) -> torch.Tensor:
    """
    LOGAN-style ensemble scoring WITHOUT passing unsupported kwargs to the backbone.
    Works with CLIP-T5-based VQAScore by calling: score_func(images=..., texts=...).

    Returns a tensor of shape (num_images,) with one LOGAN score per image.
    Higher is better.
    """
    # Base scores across paraphrases: expect (N, K) (N: images, K: prompts)
    base = score_func(images=images, texts=texts, mode=mode)
    if base.dim() == 1:                 # (N,) -> (N,1)
        base = base.unsqueeze(1)
    elif base.shape[1] != len(texts):   # some impls return (K, N)
        if base.shape[0] == len(texts) and base.shape[1] == len(images):
            base = base.t().contiguous()
        else:
            raise ValueError(f"Unexpected score shape {tuple(base.shape)}; "
                             f"expected (N,K) or (K,N).")

    # Optional: question-only prior subtraction measured on a neutral image
    if prior_image is not None:
        prior = score_func(images=[prior_image], texts=texts, mode=mode)  # (1,K) or (K,1)
        if prior.dim() == 1:
            prior = prior.unsqueeze(0)                         # (1,K)
        if prior.shape[0] == len(texts) and prior.shape[1] == 1:
            prior = prior.t()                                  # (1,K)
        base = base - prior                                    # broadcast across N

    # Optional: foil prompting (subtract score for negated/competing sense)
    if foil_texts is not None:
        assert len(foil_texts) == len(texts), "foil_texts must match texts in length"
        foil = score_func(images=images, texts=foil_texts, mode=mode)
        if foil.dim() == 1:
            foil = foil.unsqueeze(1)
        elif foil.shape[1] != len(foil_texts):
            if foil.shape[0] == len(foil_texts) and foil.shape[1] == len(images):
                foil = foil.t().contiguous()
            else:
                raise ValueError(f"Unexpected foil score shape {tuple(foil.shape)}")
        base = base - foil

    # Aggregate across paraphrases (dim=1)
    if aggregate == "harmonic":
        eps = 1e-6
        pos = torch.clamp(base, min=eps)        # conservative combine
        score = 1.0 / torch.mean(1.0 / pos, dim=1)
    else:
        score = torch.mean(base, dim=1)         # default: mean

    return score  # (N,)

