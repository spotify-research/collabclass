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

__version__ = "0.1.0"
__description__ = "Reference implementation for Maystre et al., AISTATS 2021"
__uri__ = "https://github.com/spotify-research/collabclass"
__author__ = "Lucas Maystre"
__author_email__ = "lucasm@spotify.com"
__maintainer__ = "Lucas Maystre"
__maintainer_email__ = "lucasm@spotify.com"

from .containers import (
    InteractionGraph,
    get_user_neighbors,
    get_item_neighbors,
    graph_from_edges,
)

from .algorithms import (
    cavi,
    wvrn,
    init_beta,
)

from .models import (
    sbm,
    symmetric_channel,
)

from .eval import (
    print_accuracy,
    degree_breakdown,
    degree_breakdown_topk,
    confusion_matrix,
)
