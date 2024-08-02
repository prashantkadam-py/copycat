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

from absl.testing import absltest
from absl.testing import parameterized
from langchain_community import vectorstores

from copycat.py import models
from copycat.py import testing_utils


class VertexAIEmbeddingsTest(parameterized.TestCase):

  @testing_utils.PatchEmbeddingsModel()
  def test_embed_documents_returns_embeddings(self, embeddings_model_patcher):
    model = models.BatchedVertexAIEmbeddings()
    embeddings = model.embed_documents(["test_1", "test_2"])
    self.assertLen(embeddings, 2)
    self.assertLen(embeddings[0], 768)
    self.assertLen(embeddings[1], 768)

  @testing_utils.PatchEmbeddingsModel()
  def test_embed_query_returns_embeddings(self, embeddings_model_patcher):
    model = models.BatchedVertexAIEmbeddings()
    embeddings = model.embed_query("test_1")
    self.assertLen(embeddings, 768)

  @testing_utils.PatchEmbeddingsModel()
  def test_can_be_used_as_a_rag_retriever(self, embeddings_model_patcher):
    # SKLearnVectorStore requires 20 documents minimum
    test_documents = [f"test_document_{x}" for x in range(20)]

    vectorstore = vectorstores.SKLearnVectorStore.from_texts(
        test_documents,
        embedding=models.BatchedVertexAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 1}
    )

    matched_documents = retriever.invoke("test_query")
    self.assertLen(matched_documents, 1)
    self.assertIn(matched_documents[0].page_content, test_documents)


if __name__ == "__main__":
  absltest.main()
