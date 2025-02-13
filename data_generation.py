from typing import List, Union
import random
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from prompts import prompt_dict
import numpy as np



dataset_name = "amazon"
# True: ml-1m, amazon, douban
# False: heloc, covtype
apply_encoder = True


def cluster_sample(array):
    from sklearn.cluster import KMeans
    label = KMeans(n_clusters=5).fit_predict(array)
    array["new_label"] = label
    groups = array.groupby("new_label")
    groups_out = []
    for group in groups:
        groups_out.append(group[1].drop(labels="new_label", axis=1))
    return groups_out


class LLMmodel():
    def __init__(self):
        self.prompts = prompt_dict[dataset_name]

        self.sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=3500)
        self.model = LLM(model="./llama-3-70b-instruct-awq/", trust_remote_code=True, dtype='float16',
                         tensor_parallel_size=1, enforce_eager=True, quantization="AWQ", enable_prefix_caching=True)

    def predict(self, prompt: str) -> str:
        prompts = self.prompts[0] + prompt + self.prompts[1]
        outputs = self.model.generate(prompts, self.sampling_params)
        generation = outputs[0].outputs[0].text.strip()
        return generation

model = LLMmodel()
dataset = pd.read_csv("./data/{}/{}_train.csv".format(dataset_name, dataset_name))
cols = dataset.columns
len_data = len(dataset)
groups_out = cluster_sample(dataset)


if apply_encoder:
    encoders = np.load("./data/{}/{}_encoders.npy".format(dataset_name, dataset_name), allow_pickle=True).item()
    for i in tqdm(range(0, len_data, 3)):
        prompt_id = ""
        prompt_text = ""
        for group in groups_out:
            data = group.sample(n=2)
            for idx, row in data.iterrows():
                for ind in range(len(cols)):
                    col = cols[ind]
                    if ind == (len(cols)-1):
                        prompt_text += "{} is {}, ".format(col, row[col])
                    else:
                        prompt_text += "{} is {}, ".format(col, encoders[col].inverse_transform([row[col]])[0])
                    prompt_id += "{} is {}, ".format(col, row[col])
                prompt_text += "\n"
                prompt_id += "\n"
        prompt = prompt_text + "\n\nHere are the corresponding id expressions of the above samples:\n\n" + prompt_id
        generation = model.predict(prompt_id)
        with open("./data/{}/llm_origin/".format(dataset_name) + str(i) + ".txt", "w") as fp:
            fp.write(generation)
else:
    for i in tqdm(range(1, len_data, 3)):
        prompt = ""
        for group in groups_out:
            data = group.sample(n=2)
            for idx, row in data.iterrows():
                for col in cols:
                    prompt += "{} is {}, ".format(col, row[col])
                prompt += "\n"
        generation = model.predict(prompt)
        with open("./data/{}/llm_origin/".format(dataset_name) + str(i) + ".txt", "w") as fp:
            fp.write(generation)
