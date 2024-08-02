# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text embedding model for the Copycat tool.

Contains the code to generate a text embedding from a language model.
"""

import langchain_google_vertexai

DEFAULT_VERTEX_AI_EMBEDDING_BATCH_SIZE = 5


class BatchedVertexAIEmbeddings(langchain_google_vertexai.VertexAIEmbeddings):
  """Generates embeddings from Vertex AI, which overrides the batch size.

  This is a workaround for the fact that the Vertex AI model uses batch size
  = 0 by default, which does not work well with the vectorstores, and there is
  no easy way to override it.
  """

  def embed_documents(
      self, texts: list[str], batch_size: int = 0
  ) -> list[list[float]]:
    return super().embed_documents(
        texts, batch_size=DEFAULT_VERTEX_AI_EMBEDDING_BATCH_SIZE
    )
