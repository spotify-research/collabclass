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

import numba
import numpy as np

from math import log
from scipy.special import softmax


@numba.vectorize(cache=True)
def _digamma(x):
    """Fast approximation of the digamma function, assuming x > 0."""
    # Taken from
    # <https://gist.github.com/timvieira/656d9c74ac5f82f596921aa20ecb6cc8>
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + log(x) - 0.5/x + t


@numba.njit(cache=True)
def _softmax(xs):
    """JIT-compiled implementation of the softmax function."""
    ys = np.exp(xs - max(xs))
    return ys / ys.sum()


@numba.njit(cache=True)
def cavi(graph, alpha, beta, n_iters, verbose=False):
    """CAVI algorithm for collaborative classification."""
    k = alpha.shape[1]  # Number of classes.
    alpha_ = np.copy(alpha)  # Posterior alpha.
    beta_ = np.copy(beta)  # Posterior beta.
    log_beta = np.log(beta)
    for itr in range(n_iters):
        if verbose:
            print("iteration", itr+1)
        for i in range(graph.m):
            o, s = graph.user_idx[i]
            neighbors = graph.user_edges[o:o+s]
            alpha_[i] = alpha[i] + np.sum(beta_[neighbors], axis=0)
        z1 = _digamma(alpha_)
        for j in range(graph.n):
            o, s = graph.item_idx[j]
            neighbors = graph.item_edges[o:o+s]
            beta_[j] = _softmax(log_beta[j] + np.sum(z1[neighbors], axis=0))
    return alpha_, beta_


def init_beta(k, vs, delta, prior=None):
    """Initialize `beta` matrix based on observed labels and noise level."""
    if prior is None:
        log_prior = np.zeros(k)
    else:
        log_prior = np.log(prior)
    log_beta = np.tile(log_prior, (len(vs), 1))
    log_beta[np.arange(len(vs)),vs] += (
        np.log(k - 1) + np.log(1 - delta) - np.log(delta))
    return softmax(log_beta, axis=1)


@numba.njit(cache=True)
def wvrn(graph, vs):
    """wvRN algorithm for collaborative classification."""
    k = vs.max() + 1
    xs = np.zeros((graph.m, k), dtype=np.int64)
    for i in range(graph.m):
        o, s = graph.user_idx[i]
        for j in graph.user_edges[o:o+s]:
            xs[i,vs[j]] += 1
    zs = np.zeros((graph.n, k), dtype=np.int64)
    for j in range(graph.n):
        o, s = graph.item_idx[j]
        for i in graph.item_edges[o:o+s]:
            zs[j] += xs[i]
            zs[j,vs[j]] -= 1
    out = np.zeros(graph.n, dtype=np.int64)
    for j in range(graph.n):
        out[j] = np.argmax(zs[j])
    return out
