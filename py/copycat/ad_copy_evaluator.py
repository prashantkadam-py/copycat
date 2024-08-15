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

import dataclasses

import pydantic
from sklearn.metrics import pairwise

from copycat import ad_copy_generator
from copycat import google_ads


GoogleAd = google_ads.GoogleAd
GoogleAdFormat = google_ads.GoogleAdFormat


class EvaluationResults(pydantic.BaseModel):
  """The metrics used to evaluate the generated ad."""
  headlines_are_memorised: bool | None
  descriptions_are_memorised: bool | None
  errors: list[str]
  warnings: list[str]


class EvaluationMetrics(pydantic.BaseModel):
  """Similarity between the generated ad and the training data and keywords."""

  style_similarity: float = pydantic.Field(
      ge=0.0,
      le=1.0,
      description=(
          "How similar the style of the generated ad is to the style of the"
          " training ads."
      ),
  )
  keyword_similarity: float = pydantic.Field(
      ge=0.0,
      le=1.0,
      description="How similar the generated ad is to the keywords.",
  )


@dataclasses.dataclass
class AdCopyEvaluator:
  """Evaluates the ad copy.

  Attributes:
    ad_format: The ad format.
    training_headlines: The headlines in the training data.
    training_descriptions: The descriptions in the training data.
  """

  ad_format: GoogleAdFormat
  training_headlines: set[str]
  training_descriptions: set[str]

  def has_valid_number_of_headlines(self, ad_copy: GoogleAd) -> bool:
    """Returns true if the number of headlines is valid.

    The number of headlines is valid if it is less than or equal to the maximum
    number of headlines for the ad format, and greater than or equal the minimum
    number of headlines for the ad format.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    return (ad_copy.headline_count <= self.ad_format.max_headlines) and (
        ad_copy.headline_count >= self.ad_format.min_headlines
    )

  def has_valid_number_of_descriptions(self, ad_copy: GoogleAd) -> bool:
    """Returns true if the number of descriptions is valid.

    The number of descriptions is valid if it is less than or equal to the
    maximum number of descriptions for the ad format, and greater than or equal
    to the minimum number of descriptions for the ad format.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    return (ad_copy.description_count <= self.ad_format.max_descriptions) and (
        ad_copy.description_count >= self.ad_format.min_descriptions
    )

  def has_valid_headline_lengths(self, ad_copy: GoogleAd) -> bool:
    """Returns true if the lengths of the headlines are valid.

    The lengths of the headlines are valid if they are all less than or equal to
    the maximum length for a headline.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    return all(
        len(google_ads.parse_default_dynamic_keyword_insertion(headline))
        <= self.ad_format.max_headline_length
        for headline in ad_copy.headlines
    )

  def has_valid_description_lengths(self, ad_copy: GoogleAd) -> bool:
    """Returns true if the lengths of the descriptions are valid.

    The lengths of the descriptions are valid if they are all less than or equal
    to the maximum length for a description.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    return all(
        len(google_ads.parse_default_dynamic_keyword_insertion(description))
        <= self.ad_format.max_description_length
        for description in ad_copy.descriptions
    )
    
  def has_unique_headlines(self, ad_copy: GoogleAd) -> bool:
    """Returns true if there are no duplicate headlines.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    return len(set(ad_copy.headlines)) == len(ad_copy.headlines)

  def has_unique_descriptions(self, ad_copy: GoogleAd) -> bool:
    """Returns true if there are no duplicate descriptions.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    return len(set(ad_copy.descriptions)) == len(ad_copy.descriptions)

  def is_valid(self, ad_copy: GoogleAd) -> bool:
    """Returns true if the ad copy is valid.

    This checks that the number of headlines and descriptions is within the
    limits for the ad format, the headlines and descriptions are unique, and 
    that the length of each headline and description is within the limits.

    Args:
      ad_copy: The ad copy to evaluate.
    """

    return (
        self.has_valid_number_of_headlines(ad_copy)
        and self.has_valid_number_of_descriptions(ad_copy)
        and self.has_valid_headline_lengths(ad_copy)
        and self.has_valid_description_lengths(ad_copy)
        and self.has_unique_headlines(ad_copy)
        and self.has_unique_descriptions(ad_copy)
    )

  def is_complete(self, ad_copy: GoogleAd) -> bool:
    """Returns true if the ad copy is complete.

    A complete ad copy contains the maximum number of headlines and
    descriptions.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    complete_headlines = len(ad_copy.headlines) == self.ad_format.max_headlines
    complete_descriptions = (
        len(ad_copy.descriptions) == self.ad_format.max_descriptions
    )
    return complete_headlines and complete_descriptions

  def is_underpopulated(self, ad_copy: GoogleAd) -> bool:
    """Returns true if the ad copy is underpopulated.

    This means the ad copy has fewer than the minimum number of headlines or
    descriptions.

    Args:
      ad_copy: The ad copy to evaluate.
    """
    return (len(ad_copy.headlines) < self.ad_format.min_headlines) or (
        len(ad_copy.descriptions) < self.ad_format.min_descriptions
    )

  def headlines_are_memorised(self, ad_copy: GoogleAd) -> bool:
    """Returns true if all the headlines exist in the training data."""
    if not ad_copy.headlines:
      # There are no headlines, so they cannot be memorised.
      return False

    return not (set(ad_copy.headlines) - self.training_headlines)

  def descriptions_are_memorised(self, ad_copy: GoogleAd) -> bool:
    """Returns true if all the descriptions exist in the training data."""
    if not ad_copy.descriptions:
      # There are no descriptions, so they cannot be memorised.
      return False

    return not (set(ad_copy.descriptions) - self.training_descriptions)

  def evaluate(
      self,
      ad_copy: GoogleAd,
      *,
      allow_memorised_headlines: bool = False,
      allow_memorised_descriptions: bool = False,
  ) -> EvaluationResults:
    """Evaluates the generated ad copy.

    Args:
      ad_copy: The generated ad.
      allow_memorised_headlines: Whether to allow the headlines to be memorised.
      allow_memorised_descriptions: Whether to allow the descriptions to be
        memorised.

    Returns:
      The evaluation results.
    """
    errors = []
    warnings = []

    if not self.has_valid_number_of_headlines(ad_copy):
      errors.append("Invalid number of headlines for the ad format.")
    if not self.has_valid_number_of_descriptions(ad_copy):
      errors.append("Invalid number of descriptions for the ad format.")
    if not self.has_valid_headline_lengths(ad_copy):
      errors.append("At least one headline too long for the ad format.")
    if not self.has_valid_description_lengths(ad_copy):
      errors.append("At least one description too long for the ad format.")
    if not self.has_unique_headlines(ad_copy):
      errors.append("Duplicate headlines found.")
    if not self.has_unique_descriptions(ad_copy):
      errors.append("Duplicate descriptions found.")

    headlines_are_memorised = self.headlines_are_memorised(ad_copy)
    descriptions_are_memorised = self.descriptions_are_memorised(ad_copy)

    if headlines_are_memorised:
      if allow_memorised_headlines:
        warnings.append("All headlines are memorised from the training data.")
      else:
        errors.append("All headlines are memorised from the training data.")
    if descriptions_are_memorised:
      if allow_memorised_descriptions:
        warnings.append(
            "All descriptions are memorised from the training data."
        )
      else:
        errors.append("All descriptions are memorised from the training data.")

    return EvaluationResults(
        errors=errors,
        warnings=warnings,
        headlines_are_memorised=headlines_are_memorised,
        descriptions_are_memorised=descriptions_are_memorised,
    )


def _normalize_cosine_similarity(similarity: float) -> float:
  """Converts the cosine similarity to a value between 0 and 1."""
  return min(max((1.0 + similarity) / 2.0, 0.0), 1.0)


def evaluate_ad_copy(
    *,
    google_ad: GoogleAd,
    keywords: str,
    ad_copy_vectorstore: ad_copy_generator.AdCopyVectorstore,
) -> EvaluationMetrics:
  """Evaluates the ad copy against the training data and keywords.

  This calculates two metrics:
    - The style similarity, which is how similar the style of the generated ad
      is to the style of the training ads. It is calculated by finding the 5
      most similar training ads to the generated ad, and averageing their
      similarity scores.
    - The keyword similarity, which is how similar the generated ad is to the
      keywords.

  The similarity in both cases is calculated using the cosine similarity, and
  normalising it between 0 and 1, so 0 is the least similar and 1 is the most
  similar.

  Args:
    google_ad: The generated ad.
    keywords: The keywords used to generate the ad.
    ad_copy_vectorstore: The vector store containing the training ads.

  Returns:
    The evaluation metrics.

  Raises:
    RuntimeError: If the vector store does not have an embeddings attribute.
  """
  keywords_embedding = ad_copy_vectorstore.embed_queries([keywords])[0]
  ad_embedding = ad_copy_vectorstore.embed_documents([str(google_ad)])[0]

  keyword_similarity = _normalize_cosine_similarity(
      pairwise.cosine_similarity([keywords_embedding], [ad_embedding])[0][0]
  )

  relevant_ads = ad_copy_vectorstore.get_relevant_ads([str(google_ad)], k=5)[0]
  similar_training_ad_embeddings = ad_copy_vectorstore.embed_documents(
      [str(example.google_ad) for example in relevant_ads]
  )

  style_similarity = _normalize_cosine_similarity(
      pairwise.cosine_similarity(
          similar_training_ad_embeddings, [ad_embedding]
      ).mean()
  )

  return EvaluationMetrics(
      style_similarity=style_similarity,
      keyword_similarity=keyword_similarity,
  )
