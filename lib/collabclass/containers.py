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

import collections
import numpy as np


InteractionGraph = collections.namedtuple(
    "InteractionGraph",
    (
        "m",  # Number of users.
        "n",  # Number of items.
        "user_idx",  # Mapping `i -> offset, size`.
        "item_idx",  # Mapping `j -> offset, size`.
        "user_edges",  # 1D array containing users' neighboring items.
        "item_edges",  # 1D array containing items' neighboring users.
    ))


def get_user_neighbors(graph, i):
    o, s = graph.user_idx[i]
    return graph.user_edges[o:o+s]


def get_item_neighbors(graph, j):
    o, s = graph.item_idx[j]
    return graph.item_edges[o:o+s]


def graph_from_edges(m, n, edges):
    u2v = collections.defaultdict(list)
    v2u = collections.defaultdict(list)
    for i, j in edges:
        u2v[i].append(j)
        v2u[j].append(i)
    user_idx = np.zeros((m, 2), dtype=int)
    user_edges = np.zeros(len(edges), dtype=int)
    offset = 0
    for i in range(m):
        s = len(u2v[i])
        user_idx[i,:] = (offset, s)
        user_edges[offset:offset+s] = u2v[i]
        offset += s
    item_idx = np.zeros((n, 2), dtype=int)
    item_edges = np.zeros(len(edges), dtype=int)
    offset = 0
    for j in range(n):
        s = len(v2u[j])
        item_idx[j,:] = (offset, s)
        item_edges[offset:offset+s] = v2u[j]
        offset += s
    return InteractionGraph(
        m=m,
        n=n,
        user_idx=user_idx,
        item_idx=item_idx,
        user_edges=user_edges,
        item_edges=item_edges)
