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

import asyncio
from collections.abc import Sequence
import dataclasses
import enum
import json
import pathlib
from typing import Any, AsyncIterable, Coroutine, Hashable, TypeVar

from google.cloud.aiplatform.vertexai import generative_models
from langchain_community import vectorstores as community_vectorstores
import numpy as np
import pandas as pd
import pydantic

from copycat.py import google_ads
from copycat.py import models as embedding_models


AsyncGenerationResponse = Coroutine[
    Any,
    Any,
    generative_models.GenerationResponse
    | AsyncIterable[generative_models.GenerationResponse],
]

GoogleAd = google_ads.GoogleAd
GoogleAdFormat = google_ads.GoogleAdFormat

ValidationError = pydantic.ValidationError
FinishReason = generative_models.FinishReason

SafetySettingsType = (
    dict[generative_models.HarmCategory, generative_models.HarmBlockThreshold]
    | list[generative_models.SafetySetting]
)


SKLEARN_VECTORSTORE_FILE_NAME = "vectorstore"
VECTORSTORE_PARAMS_FILE_NAME = "vectorstore_params.json"


class ModelName(enum.Enum):
  GEMINI_1_0_PRO = "gemini-pro"
  GEMINI_1_5_PRO = "gemini-1.5-pro-preview-0514"
  GEMINI_1_5_FLASH = "gemini-1.5-flash-preview-0514"


class EmbeddingModelName(enum.Enum):
  TEXT_EMBEDDING_GECKO = "textembedding-gecko"
  TEXT_EMBEDDINGS = "text-embedding-004"
  TEXT_EMBEDDING_GECKO_MULTILINGUAL = "textembedding-gecko-multilingual"
  TEXT_EMBEDDING_MULTILINGUAL = "text-multilingual-embedding-002"


class TextGenerationRequest(pydantic.BaseModel):
  """The request to generate text."""

  system_instruction: str
  prompt: list[generative_models.Content]
  chat_model_name: ModelName
  temperature: float
  top_k: int
  top_p: float
  safety_settings: SafetySettingsType | None

  class Config:
    arbitrary_types_allowed = True

  def to_markdown(self):
    lines = [
        "**Model Parameters:**",
        f"Model name: {self.chat_model_name.value}",
        f"Temperature: {self.temperature}",
        f"Top K: {self.top_k}",
        f"Top P: {self.top_p}",
        f"Safety settings: {self.safety_settings}",
        "**System instruction:**",
        self.system_instruction,
    ]

    for content in self.prompt:
      lines.append(f"**{content.role.title()}:**")
      lines.append(content.parts[0].text)

    return "\n\n".join(lines)


@dataclasses.dataclass
class AdCopyVectorstore:
  """The vector store containing the ad copies.

  Each record contains both a text that will be matched to queries and some
  metadata. The text is either a headline or a description that exists in the
  ad, and the metadata contains the full list of headlines, descriptions and
  keywords for that ad. Each ad will appear in the vectorstore multiple times,
  once for each headline and description it uses. This allows the ads to be
  matched to the query based on the most relavent individual headline or
  description, rather than an average over all of them.

  Attributes:
    embedding_model_name: The name of the embedding model to use.
    vectorstore: The vector store containing the ad copies.
    persist_path: The path to persist the vector store to.
    max_fetch_k: The maximum number of documents to fetch before performing mmr
      search.
  """

  embedding_model_name: EmbeddingModelName
  vectorstore: community_vectorstores.SKLearnVectorStore
  persist_path: str

  @classmethod
  def _deduplicate_ads(cls, data: pd.DataFrame) -> pd.DataFrame:
    """Deduplicates the ads in the training data.

    If the same ads are used for multiple sets of keywords, select just one
    random keywords set for each ad. We don't need to have identical ads in
    the vectorstore.

    Args:
      data: The training data containing the headlines, descriptions and
        keywords.

    Returns:
      The deduplicated training data.
    """
    data = data.copy()
    data["headlines"] = data["headlines"].apply(tuple)
    data["descriptions"] = data["descriptions"].apply(tuple)
    data = (
        data.groupby(["headlines", "descriptions"], group_keys=False)
        .sample(1)
        .reset_index(drop=True)
    )
    data["headlines"] = data["headlines"].apply(list)
    data["descriptions"] = data["descriptions"].apply(list)
    return data

  @classmethod
  def _explode_headlines_and_descriptions(
      cls, data: pd.DataFrame
  ) -> pd.DataFrame:
    """Explodes the headlines and descriptions in the training data.

    Creates a new column called "exploded_headlines_and_descriptions" which
    contains all of the headlines and descriptions from the original columns.
    These exploded headlines and descriptions are then used as the documents
    in the vectorstore.

    For example, if the original data contained a single row with:

      - headlines: ["headline 1", "headline 2"]
      - descriptions: ["description 1", "description 2"]

    Then this row would be exploded into 4 rows, with the following values:

      - headlines:                            ["headline 1", "headline 2"]
      - descriptions:                         ["description 1", "description 2"]
      - exploded_headlines_and_descriptions:  "headline 1"

      - headlines:                            ["headline 1", "headline 2"]
      - descriptions:                         ["description 1", "description 2"]
      - exploded_headlines_and_descriptions:  "headline 2"

      - headlines:                            ["headline 1", "headline 2"]
      - descriptions:                         ["description 1", "description 2"]
      - exploded_headlines_and_descriptions:  "description 1"

      - headlines:                            ["headline 1", "headline 2"]
      - descriptions:                         ["description 1", "description 2"]
      - exploded_headlines_and_descriptions:  "description 2"

    Args:
      data: The training data containing the headlines, descriptions and
        keywords.

    Returns:
      The training data with the exploded headlines and descriptions.
    """
    headlines = data["headlines"].explode()
    descriptions = data["descriptions"].explode()

    exploded_headlines_and_descriptions = (
        pd.concat([headlines, descriptions])
        .reset_index()
        .drop_duplicates()
        .set_index("index")
        .rename(columns={0: "exploded_headlines_and_descriptions"})
    )
    exploded_data = data.merge(
        exploded_headlines_and_descriptions, left_index=True, right_index=True
    )
    return exploded_data

  @classmethod
  def create_from_pandas(
      cls,
      training_data: pd.DataFrame,
      embedding_model_name: str | EmbeddingModelName,
      persist_path: str,
  ) -> "AdCopyVectorstore":
    """Creates a vector store containing the ad copies from pandas.

    The vectorstore is created from the provided training data. The training
    data contains the real ad copies and keywords they were used for. Make sure
    the ad copy is high quality as this is what the model will learn from.

    The training_data must contain the following columns:
      - headlines: The headlines of the ad copy. This should be a list of
        strings.
      - descriptions: The descriptions of the ad copy. This should be a list of
        strings.
      - keywords: The keywords the ad copy was used for. This should be a
        string of comma separated keywords.

    Args:
      training_data: The training data containing the real ad copies and
        keywords.
      embedding_model_name: The name of the embedding model to use.
      persist_path: The path to persist the vector store to.

    Returns:
      A sklearn vector store containing the ad copies.
    """
    embedding_model_name = EmbeddingModelName(embedding_model_name)
    embedding_model = embedding_models.BatchedVertexAIEmbeddings(
        model_name=embedding_model_name.value
    )

    data = (
        training_data[["headlines", "descriptions", "keywords"]]
        .copy()
        .pipe(cls._deduplicate_ads)
        .pipe(cls._explode_headlines_and_descriptions)
    )

    metadata = data[["headlines", "descriptions", "keywords"]].to_dict(
        "records"
    )
    texts = data["exploded_headlines_and_descriptions"].values.tolist()

    path = pathlib.Path(persist_path)

    return cls(
        vectorstore=community_vectorstores.SKLearnVectorStore.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadata,
            persist_path=str(path / SKLEARN_VECTORSTORE_FILE_NAME),
            serializer="parquet",
        ),
        persist_path=persist_path,
        embedding_model_name=embedding_model_name,
    )

  @classmethod
  def load(cls, persist_path: str) -> "AdCopyVectorstore":
    """Loads the vectorstore from the provided path."""
    path = pathlib.Path(persist_path)

    with open(path / VECTORSTORE_PARAMS_FILE_NAME, "r") as f:
      params = json.load(f)

    embedding_model = embedding_models.BatchedVertexAIEmbeddings(
        model_name=params["embedding_model_name"]
    )
    params["vectorstore"] = community_vectorstores.SKLearnVectorStore(
        embedding=embedding_model,
        persist_path=str(path / SKLEARN_VECTORSTORE_FILE_NAME),
        serializer="parquet",
    )
    params["persist_path"] = persist_path
    params["embedding_model_name"] = EmbeddingModelName(
        params["embedding_model_name"]
    )
    return cls(**params)

  def write(self) -> None:
    """Loads the vectorstore from the provided path."""
    path = pathlib.Path(self.persist_path)
    path.mkdir(parents=True, exist_ok=True)

    params = {}
    params["embedding_model_name"] = self.embedding_model_name.value

    with open(path / VECTORSTORE_PARAMS_FILE_NAME, "w") as f:
      json.dump(params, f)

    self.vectorstore.persist()

  @property
  def max_fetch_k(self) -> int:
    return len(self.vectorstore._ids)

  def get_relevant_ads(
      self, query: str, k: int, fetch_k: int = 1000, lambda_mult: float = 0.5
  ) -> list[tuple[str, GoogleAd]]:
    """Returns the k most relevant ads for the provided query.

    The ads are retrieved from the vectorstore using the provided query. The
    ads are then filtered using maximal marginal relevance (MMR) to return the
    k most relevant ads.

    Args:
      query: The query to use to retrieve the ads.
      k: The number of ads to return.
      fetch_k: The number of ads to fetch when constructing the prompt, before
        filtering based on maximum marginal relevance.
      lambda_mult: The lambda multiplier to use when filtering the in-context
        examples. Controls the trade-off between similarity to the query and
        similarity to the other examples. Must be between 0 (most variety in the
        examples) to 1 (most similar to the query).

    Returns:
      The k most relavent pairs of keywords and ads.
    """
    fetch_k = min(self.max_fetch_k, fetch_k)

    query_embedding = np.asarray(self.vectorstore.embeddings.embed_query(query))
    similar_documents_with_duplicates = self.vectorstore.similarity_search(
        query, k=fetch_k
    )

    similar_ads = list(
        set([
            (
                doc.metadata["keywords"],
                google_ads.GoogleAd(
                    headlines=doc.metadata["headlines"],
                    descriptions=doc.metadata["descriptions"],
                ),
            )
            for doc in similar_documents_with_duplicates
        ])
    )

    document_embeddings = self.vectorstore.embeddings.embed_documents(
        list(map(lambda x: str(x[1]), similar_ads))
    )

    idx = community_vectorstores.utils.maximal_marginal_relevance(
        query_embedding, document_embeddings, lambda_mult=lambda_mult, k=k
    )

    return [similar_ads[i] for i in idx]


def _construct_new_ad_copy_user_message(
    keywords: str,
    keywords_specific_instructions: str = "",
) -> generative_models.Content:
  """Constructs the json content."""
  content = ""
  if keywords_specific_instructions:
    content += (
        "For the next set of keywords, please consider the following additional"
        f" instructions:\n\n{keywords_specific_instructions}\n\n"
    )
  content += f"Keywords: {keywords}"

  return generative_models.Content(
      role="user",
      parts=[generative_models.Part.from_text(content)],
  )


def construct_examples_for_new_ad_copy_generation(
    ad_documents: list[tuple[str, GoogleAd]],
) -> list[generative_models.Content]:
  """Creates the in-context examples for new ad copy generation.

  Args:
    ad_documents: The list of keywords and ad pairs to convert, sorted in order
      of relevance.

  Returns:
    A list of in-context examples for new ad copy generation. This is a list of
    messages, alternating between the keywords and expected response from each
    ad document. The expected response is a json string containing the headlines
    and descriptions of the ad copy. The messages are sorted so that the most
    relevant examples are last. This ensures the model see's the most relevant
    examples last, making them more likely to influence the model's output.
  """
  in_context_examples = []
  for keywords, ad in reversed(ad_documents):
    in_context_examples.append(_construct_new_ad_copy_user_message(keywords))
    in_context_examples.append(
        generative_models.Content(
            role="model",
            parts=[generative_models.Part.from_text(ad.model_dump_json())],
        )
    )

  return in_context_examples


def construct_system_instruction(
    system_instruction: str,
    style_guide: str,
    system_instruction_kwargs: dict[str, Any],
) -> str:
  """Constructs the system instruction by adding the style guide and kwargs.

  Args:
    system_instruction: The system instruction to use. This should explain the
      task to the model.
    style_guide: The style guide to use.
    system_instruction_kwargs: The keyword arguments are used to replace any
      placeholders in the system prompt.

  Returns:
  The formatted system prompt.
  """
  if style_guide:
    system_instruction += "\n\n" + style_guide
  if system_instruction_kwargs:
    system_instruction = system_instruction.format(**system_instruction_kwargs)
  return system_instruction


def construct_new_ad_copy_prompt(
    in_context_example_content: list[generative_models.Content],
    keywords: str,
    keywords_specific_instructions: str = "",
) -> list[generative_models.Content]:
  """Constructs the full copycat prompt for generating new ad copy.

  The prompt consists of the system prompt, the in-context examples, and the
  human message containing the keywords. It also includes instructions on how
  to format the output.

  Args:
    in_context_example_content: The in-context examples to use.
    keywords: The keywords to generate the ad copy for.
    keywords_specific_instructions: Any additional context to use for the new
      keywords. This could include things like information from the landing
      page, information about specific discounts or promotions, or any other
      relevant information.

  Returns:
    A list of Content representing the prompt.
  """
  content_specific_instructions = _construct_new_ad_copy_user_message(
      keywords, keywords_specific_instructions
  )
  return in_context_example_content + [content_specific_instructions]


HashableTypeVar = TypeVar("HashableTypeVar", bound=Hashable)


def _deduplicate_list_keep_order(
    seq: Sequence[HashableTypeVar],
) -> list[HashableTypeVar]:
  seen = set()
  seen_add = seen.add
  return [x for x in seq if not (x in seen or seen_add(x))]


def remove_invalid_headlines_and_descriptions(
    google_ad: GoogleAd, google_ad_format: GoogleAdFormat
) -> None:
  """Removes invalid headlines and descriptions from the ad.

  First it removes any duplicate headlines or descriptions, then removes any
  headlines or descriptions that are too long. Then it removes any headlines or
  descriptions that are not in the first k headlines or descriptions.

  Args:
    google_ad: The ad to remove the invalid headlines and descriptions from.
    google_ad_format: The format of the ad.
  """
  google_ad.headlines = _deduplicate_list_keep_order(google_ad.headlines)
  google_ad.descriptions = _deduplicate_list_keep_order(google_ad.descriptions)

  google_ad.headlines = [
      headline
      for headline in google_ad.headlines
      if len(google_ads.parse_default_dynamic_keyword_insertion(headline))
      <= google_ad_format.max_headline_length
  ]
  google_ad.descriptions = [
      description
      for description in google_ad.descriptions
      if len(google_ads.parse_default_dynamic_keyword_insertion(description))
      <= google_ad_format.max_description_length
  ]

  if len(google_ad.headlines) > google_ad_format.max_headlines:
    google_ad.headlines = google_ad.headlines[: google_ad_format.max_headlines]
  if len(google_ad.descriptions) > google_ad_format.max_descriptions:
    google_ad.descriptions = google_ad.descriptions[
        : google_ad_format.max_descriptions
    ]


def _format_instructions(output_schema: type[pydantic.BaseModel]) -> str:
  """Returns the output schema as a string to be used in the prompt."""
  elements = []
  for k, v in output_schema.model_fields.items():
    elements.append(f"'{k}': {v.annotation}")
  element_lines = ",".join(map(lambda x: "\n  " + x, elements))
  return (
      f"Return: {output_schema.__name__}\n{output_schema.__name__} = "
      + "{"
      + element_lines
      + "\n}"
  )


def async_generate_google_ad_json(
    request: TextGenerationRequest,
) -> AsyncGenerationResponse:
  """Generates a GoogleAd from the text generation request asynchronously.

  This function ensures that the generated response is a valid json
  representation of a GoogleAd, by appending formatting instructions to the
  system instruction and including a response schema in the generation config
  for models that accept it.

  Args:
    request: The text generation request, containing the prompt, system
      instruction, style guide, and other parameters.

  Returns:
    The generated response, which is a valid json representation of a GoogleAd.
  """

  model_name = ModelName(request.chat_model_name)

  generation_config_params = dict(
      temperature=request.temperature,
      top_k=request.top_k,
      top_p=request.top_p,
      response_mime_type="application/json",
  )

  if model_name is ModelName.GEMINI_1_5_PRO:
    # Gemini 1.5 pro supports constrained generation, which allows the schema
    # to be passed as an arguments to the generation config.
    response_schema = GoogleAd.model_json_schema()
    response_schema["description"] = (
        response_schema.pop("description").replace("\n", " ").replace("  ", " ")
    )
    generation_config_params["response_schema"] = response_schema

  generation_config = generative_models.GenerationConfig(
      **generation_config_params
  )

  system_instruction = (
      f"{request.system_instruction}\n\n{_format_instructions(GoogleAd)}"
  )

  model = generative_models.GenerativeModel(
      model_name=model_name.value,
      generation_config=generation_config,
      system_instruction=system_instruction,
      safety_settings=request.safety_settings,
  )

  response = model.generate_content_async(request.prompt)

  return response


def generate_google_ad_json_batch(
    requests: list[TextGenerationRequest],
) -> list[generative_models.GenerationResponse]:
  """Generates a GoogleAd from the provided text generation request.

  This function ensures that the generated response is a valid json
  representation of a GoogleAd, by appending formatting instructions to the
  system instruction and including a response schema in the generation config
  for models that accept it.

  Args:
    requests: A list of text generation requests, containing the prompts, system
      instructions, style guides, and other parameters.

  Returns:
    The generated responses, which are valid json representations of GoogleAds.

  Raises:
    RuntimeError: If one of the responses is not a valid json representation of
    a GoogleAd. This shouldn't happen unless the gemini api changes.
  """
  loop = asyncio.get_event_loop()
  outputs = loop.run_until_complete(
      asyncio.gather(*list(map(async_generate_google_ad_json, requests)))
  )
  for output in outputs:
    if not isinstance(output, generative_models.GenerationResponse):
      raise RuntimeError(
          "One of the responses is not a GenerationResponse. Instead got:"
          f" {output}"
      )

  return outputs
