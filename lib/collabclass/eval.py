# Copyright 2021 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt


def print_accuracy(vs_est, vs, vs_hat):
    n = len(vs)
    correct = (vs_est == vs)
    corrupted = (vs_hat != vs)
    acc = np.count_nonzero(correct) / n
    fpr = np.count_nonzero(~correct[~corrupted]) / np.count_nonzero(~corrupted)
    fnr = np.count_nonzero(~correct[corrupted]) / np.count_nonzero(corrupted)
    print(f"accuracy:   {acc:.4f}")
    print(f"error rate: {(1 - acc):.4f}")
    print(f"FP rate:    {fpr:.4f}")
    print(f"FN rate:    {fnr:.4f}")

    
def degree_breakdown(vs_est, vs, graph, qs=(50, 90, 95)):
    ps = np.percentile(graph.item_idx[:,1], qs)
    correct = (vs_est == vs)
    for p, q in zip(ps, qs):
        mask = (graph.item_idx[:,1] >= p)
        acc = np.count_nonzero(correct[mask]) / np.count_nonzero(mask)
        print(f"accuracy at {q}th percentile (d ≥ {p:.0f}): {acc:.4f}")


def degree_breakdown_topk(scores, vs, graph, k=2, qs=(0, 50, 90, 95)):
    ps = np.percentile(graph.item_idx[:,1], qs)
    rankings = np.argsort(scores, axis=1)[:,::-1]
    topk = np.zeros(len(vs), dtype=bool)
    for i in range(k):
        topk |= (rankings[:,i] == vs)
    for p, q in zip(ps, qs):
        mask = (graph.item_idx[:,1] >= p)
        acc = np.count_nonzero(topk[mask]) / np.count_nonzero(mask)
        print(f"top-{k} accuracy at {q}th percentile (d ≥ {p:.0f}): {acc:.4f}")
    
    
def confusion_matrix(vs_est, vs, plot=True):
    ks = np.array(sorted(set(vs) | set(vs_est)))
    mat = np.zeros((len(ks), len(ks)))
    for a, b in zip(vs, vs_est):
        mat[a, b] += 1
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.log(mat))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(ks)
        ax.set_yticks(ks)
    return mat
