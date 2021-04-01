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

from .containers import graph_from_edges


def sbm(m, n, k, s, alpha):
    """Sparse & biased interaction model."""
    # Draw a class for each item.
    vs = np.random.choice(k, size=n, replace=True)
    items_by_class = {cls: np.nonzero(vs == cls)[0] for cls in range(k)}
    # Draw a distribution for each user.
    us = np.random.dirichlet(alpha * np.ones(k), size=m)
    edges = list()
    for i in range(m):
        # Draw classes from user-specific class proportions.
        choices = np.random.multinomial(s, us[i])
        for cls, num in enumerate(choices):
            # Draw items uniformly at random from class `cls`.
            items = np.random.choice(
                items_by_class[cls], size=num, replace=False)
            edges.extend((i, j) for j in items)
    graph = graph_from_edges(m, n, edges)
    return (us, vs, graph)


def symmetric_channel(vs, k, delta):
    """k-ary symmetric channel with error probability `delta`."""
    out = np.copy(vs)
    zs = np.random.rand(len(vs)) < delta * (1 + 1 / (k - 1))
    out[zs] = np.random.choice(k, size=np.count_nonzero(zs))
    return out
