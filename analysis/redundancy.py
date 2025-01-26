import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertModel
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from difflib import SequenceMatcher
import json
from matplotlib import pyplot as plt
import sys
import torch
import time
import os
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

def find_openion(text):
    match = re.search(r"Your Opinion:(.*?)(?=\n\n|$)", text, re.DOTALL)
    your_opinion = match.group(1).strip() if match else "No 'Your Opinion' section found."
    return your_opinion



def lexical_similarity(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = np.array(cosine_similarity(tfidf_matrix))
    # upper_triangular = np.triu(similarity_matrix, k=1)
    cossim = np.mean(similarity_matrix)

    vectorizer = CountVectorizer(binary=True, tokenizer=lambda x: x.split())
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    token_sets = [set(feature_names[i] for i in X[row].nonzero()[1]) for row in range(X.shape[0])]
    # breakpoint()
    # print(upper_triangular)
    def jaccard_similarity(set1, set2):
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 1.0
    n = len(token_sets)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = jaccard_similarity(token_sets[i], token_sets[j])

    # upper_triangular = np.triu(similarity_matrix, k=1)
    jaccard = np.mean(similarity_matrix)
    return cossim, jaccard


def semantic_similarity(corpus):
    # breakpoint()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoding = tokenizer(corpus, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoding)
        embeddings = output.last_hidden_state[:, -1]
    cossim = cosine_similarity(embeddings)
    mean_cossim = np.mean(cossim)
    return mean_cossim

        

def mutual_information(corpus):
    avg_mutual_info = 0
    for sent1 in corpus:
        for sent2 in corpus: 
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform([sent1, sent2]).toarray()
                joint_counts = np.dot(X[0].reshape(-1, 1), X[1].reshape(1, -1))
                joint_prob = joint_counts / joint_counts.sum()
                P_x = joint_prob.sum(axis=1)
                P_y = joint_prob.sum(axis=0)
                mutual_info = 0
                for i in range(len(P_x)):
                    for j in range(len(P_y)):
                        if joint_prob[i, j] > 0:
                            mutual_info += joint_prob[i, j] * np.log(joint_prob[i, j] / (P_x[i] * P_y[j]))
                avg_mutual_info += mutual_info
    avg_mutual_info /= (len(corpus)**2)
    return avg_mutual_info
        
def overlap(corpus):
    average_overlap = 0
    for sent1 in corpus:
        for sent2 in corpus:
            if sent1!=sent2:
                sim = SequenceMatcher(None, sent1, sent2).ratio()
                average_overlap += sim
    return average_overlap/(len(corpus)**2)

            


def ConvGraph():
    return


if __name__ == "__main__":
    # take file path from sys args
    # breakpoint()
    dir_path = sys.argv[1]

    # take all the files starting with improv in the directory and make a list of them
    # breakpoint()
    files = []
    for file in os.listdir(dir_path):
        if file.startswith("improv"):
            files.append(file)
    # breakpoint()
    for file in tqdm(files):
        # remove the extension to get the filename
        # start = time.time()
        cossim_full = []
        jaccard_full = []
        semsim_full = []
        mutual_info_full = []
        overlap_full = []
        file_name_no_ext = file.split(".")[0]
        full_file_path = os.path.join(dir_path, file)
        
        with open(full_file_path) as f:
            i = 0
            for line in f:
                data = json.loads(line)
                conversation = []
                # conversation.append(data['gyneacologist'])
                # conversation.append(data['oncologist'])
                # conversation.append(data['neurologist'])
                # conversation.append(data['cardiologist'])
                # conversation.append(data['endocrinologist'])
                d1 = data['doctor1']
                d2 = data['doctor2']
                d3 = data['doctor3']
                d4 = data['doctor4']
                d5 = data['doctor5']
                
                dr1_openion = find_openion(d1)
                dr2_openion = find_openion(d2)
                dr3_openion = find_openion(d3)
                dr4_openion = find_openion(d4)
                dr5_openion = find_openion(d5)
                # breakpoint()

                conversation.append(dr1_openion)
                conversation.append(dr2_openion)
                conversation.append(dr3_openion)
                conversation.append(dr4_openion)
                conversation.append(dr5_openion)

                #analysis of the conversation
                cossim, jaccard = lexical_similarity(conversation)
                # print(f"Lexical Similarity for {i} done")
                semsim = semantic_similarity(conversation)
                # print(f"Semantic Similarity for {i} done")
                mutual_info = mutual_information(conversation)
                # print(f"Mutual Information for {i} done")
                av_overlap = overlap(conversation)
                # print(f"Overlap for {i} done")

                cossim_full.append(cossim)
                jaccard_full.append(jaccard)
                semsim_full.append(semsim)
                mutual_info_full.append(mutual_info)
                overlap_full.append(av_overlap)
                i += 1
        
        
        # end = time.time()
        # print(f"Time taken for analysis: {end-start}")

        # make histograms for each analysis

        fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharey=True)

        # Plot histograms
        axes[0,0].hist(cossim_full, bins=100, edgecolor='black')
        axes[0,0].set_title("Cosine Similarity TfIdf")

        axes[0,1].hist(jaccard_full, bins=100, edgecolor='black')
        axes[0,1].set_title("Jaccard Similarity")

        axes[2,0].hist(semsim_full, bins=100, edgecolor='black')
        axes[2,0].set_title("Cosine Similarity Bert")

        axes[1,0].hist(mutual_info_full, bins=100, edgecolor='black')
        axes[1,0].set_title("Mutual Information")

        axes[1,1].hist(overlap_full, bins=100, edgecolor='black')
        axes[1,1].set_title("Overlap")

        # Add overall labels
        fig.suptitle(f"{file_name_no_ext} Analysis")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # Show the plot
        plt.tight_layout()
        plt.show()

        # save the plot
        plt.savefig(f"{dir_path}/plots/{file_name_no_ext}_analysis.png")




    