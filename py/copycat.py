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

"""The main module for Copycat.

Contains the code to generate copycat ad copies.
"""

import dataclasses
import json
import pathlib
import warnings
from typing import Any

import pandas as pd
import pydantic

from copycat.py import ad_copy_evaluator
from copycat.py import ad_copy_generator
from copycat.py import google_ads

GoogleAd = google_ads.GoogleAd
GoogleAdFormat = google_ads.GoogleAdFormat
EvaluationMetrics = ad_copy_evaluator.EvaluationMetrics
ValidationError = pydantic.ValidationError
ModelName = ad_copy_generator.ModelName
EmbeddingModelName = ad_copy_generator.EmbeddingModelName
TextGenerationRequest = ad_copy_generator.TextGenerationRequest

# Below are not used in this file, they are included for the user to easily
# adjust the safety settings in copycat without having to import
# generative_models from vertex ai.
HarmCategory = ad_copy_generator.generative_models.HarmCategory
HarmBlockThreshold = ad_copy_generator.generative_models.HarmBlockThreshold
ALL_SAFETY_SETTINGS_OFF = {
    harm_category: HarmBlockThreshold.BLOCK_NONE
    for harm_category in HarmCategory
}


DEFAULT_SYSTEM_INSTRUCTION = """\
You are an expert marketing professional, working for {company_name}. You are
tasked with writing new headlines and descriptions for Google Ads in {language},
given a new set of keywords, that will maximize engagement and clicks.
Keywords are words or phrases that are used to match ads with the terms that
people are searching for, so the copy should be engaging for someone searching
for those keywords. Each ad must have a list of {max_headlines} headlines 
and a list of {max_descriptions} descriptions. Each headline must be no 
longer than 30 characters, and each description must be no longer than 90 
characters.
""".replace("\n", " ").replace("  ", " ")

PARAMS_FILE_NAME = "params.json"
AD_COPY_VECTORSTORE_FILE_NAME = "ad_copy_vectorstore"


class CopycatResponse(pydantic.BaseModel):
  """The response from Copycat.

  Attributes:
    google_ad: The generated ad.
    keywords: The keywords used to generate the ad, as a comma separated string.
    headline_is_memorised: Whether the headline is memorised.
    description_is_memorised: Whether the description is memorised.
    success: Whether the generation was successful.
    error_message: The error message if the generation was not successful.
    evaluation_metrics: The metrics used to evaluate the model. Defaults to None
      if the evaluation has not been performed.
  """

  google_ad: GoogleAd
  keywords: str
  headlines_are_memorised: bool | None
  descriptions_are_memorised: bool | None
  error_message: str
  evaluation_metrics: EvaluationMetrics | None = None

  @property
  def success(self) -> bool:
    return not self.error_message


@dataclasses.dataclass
class Copycat:
  """The Copycat model which generates ad copies in the advertisers style.

  Attributes:
    ad_copy_vectorstore: The vectorstore containing the training ad copies.
    ad_format: The ad format that copycat will generate (same as the ad format
      of the examples in the vectorstore).
    unique_headlines: The unique headlines from the training data.
    unique_descriptions: The unique descriptions from the training data.
    persist_path: The path to persist the model to.
    output_parser: The output parser to use to parse the output of the chat
      model to a GoogleAd.
    ad_copy_evaluator: The ad copy evaluator to use to evaluate the generated ad
      copies.
  """

  ad_copy_vectorstore: ad_copy_generator.AdCopyVectorstore
  ad_format: GoogleAdFormat
  unique_headlines: set[str]
  unique_descriptions: set[str]
  persist_path: str

  @property
  def ad_copy_evaluator(self) -> ad_copy_evaluator.AdCopyEvaluator:
    return ad_copy_evaluator.AdCopyEvaluator(
        self.ad_format,
        training_headlines=self.unique_headlines,
        training_descriptions=self.unique_descriptions,
    )

  @classmethod
  def _clean_invalid_ads(
      cls, data: pd.DataFrame, ad_format: GoogleAdFormat, on_invalid_ad: str
  ) -> pd.DataFrame:
    """Cleans the invalid ads from the training data.

    Args:
      data: The training data containing the headlines, descriptions and
        keywords.
      ad_format: The ad format used in this vectorstore.
      on_invalid_ad: The action to take on invalid ads. Must be one of "raise",
        "skip", or "drop".

    Returns:
      The training data with the invalid ads handled. If on_invalid_ad is
      "raise", then an error is raised. If on_invalid_ad is "skip", then the
      invalid ads are kept in the training data. If on_invalid_ad is "drop",
      then the invalid ads are dropped from the training data.

    Raises:
      ValueError: If on_invalid_ad is not one of "raise", "skip", or "drop".
      ValueError: If there are invalid ads in the training data and
        on_invalid_ad is "raise".
    """
    evaluator = ad_copy_evaluator.AdCopyEvaluator(
        ad_format, training_headlines=set(), training_descriptions=set()
    )

    if on_invalid_ad not in ["raise", "skip", "drop"]:
      raise ValueError(
          f"Invalid value for on_invalid_ad: {on_invalid_ad}. Must be one of"
          " 'raise', 'skip', or 'drop'."
      )

    is_invalid = data.apply(
        lambda row: not evaluator.is_valid(
            GoogleAd(
                headlines=row["headlines"], descriptions=row["descriptions"]
            )
        ),
        axis=1,
    )
    n_invalid_ads = is_invalid.sum()
    frac_invalid_ads = n_invalid_ads / len(data)
    error_message = (
        f"{n_invalid_ads:,} ({frac_invalid_ads:.2%}) invalid ads found in the"
        " training data."
    )

    if n_invalid_ads > 0:
      if on_invalid_ad == "raise":
        raise ValueError(error_message)
      elif on_invalid_ad == "skip":
        warnings.warn(error_message + " Keeping them in the training data.")
      elif on_invalid_ad == "drop":
        warnings.warn(error_message + " Dropping them from the training data.")
        data = data[~is_invalid]

    return data

  @classmethod
  def create_from_pandas(
      cls,
      training_data: pd.DataFrame,
      embedding_model_name: str | EmbeddingModelName,
      persist_path: str,
      ad_format: str | GoogleAdFormat,
      on_invalid_ad: str = "drop",
  ) -> "Copycat":
    """Creates a Copycat model from a pandas dataframe.

    The pandas dataframe must contain the columns "headline", "description", and
    "keywords", with a different row per ad.

    Args:
      training_data: The historical ad copies to learn the style from. Must
        contain the columns "headline", "description", and "keywords".
      embedding_model_name: The name of the embedding model to use to create the
        ad copy vectorstore.
      persist_path: The path to persist the model to.
      ad_format: The ad format that copycat will generate (same as the ad format
        of the examples in the training data).
      on_invalid_ad: How to handle invalid ads in the training data. Must be one
        of "drop", "raise", or "skip". "drop" means that the invalid ads will be
        dropped. "raise" means that an exception will be raised. "skip" means
        that the invalid ads will remain in the training data.

    Returns:
      A Copycat model.

    Raises:
      ValueError: If the training data does not contain the required columns.
    """
    if isinstance(ad_format, str):
      ad_format = google_ads.get_google_ad_format(ad_format)

    required_columns = {"headlines", "descriptions", "keywords"}
    missing_columns = required_columns - set(training_data.columns)
    if missing_columns:
      raise ValueError(
          f"Training data must contain the columns {sorted(required_columns)}."
          f" Missing columns: {sorted(missing_columns)}."
      )

    training_data = cls._clean_invalid_ads(
        training_data, ad_format, on_invalid_ad
    )

    ad_copy_vectorstore = (
        ad_copy_generator.AdCopyVectorstore.create_from_pandas(
            training_data=training_data,
            embedding_model_name=embedding_model_name,
            persist_path=persist_path + "/" + AD_COPY_VECTORSTORE_FILE_NAME,
        )
    )

    unique_headlines = set(
        training_data["headlines"].explode().unique().tolist()
    )
    unique_descriptions = set(
        training_data["descriptions"].explode().unique().tolist()
    )

    return cls(
        ad_copy_vectorstore=ad_copy_vectorstore,
        unique_headlines=unique_headlines,
        unique_descriptions=unique_descriptions,
        persist_path=persist_path,
        ad_format=ad_format,
    )

  @classmethod
  def load(cls, persist_path: str) -> "Copycat":
    """Loads an existing Copycat model from a file.

    Args:
      persist_path: The path to the file containing the Copycat model.

    Returns:
      A Copycat model.
    """
    path = pathlib.Path(persist_path)

    with open(path / PARAMS_FILE_NAME) as f:
      params = json.load(f)

    params["ad_format"] = GoogleAdFormat(**params.pop("ad_format_params"))
    params["unique_headlines"] = set(params["unique_headlines"])
    params["unique_descriptions"] = set(params["unique_descriptions"])

    ad_copy_vectorstore = ad_copy_generator.AdCopyVectorstore.load(
        str(path / AD_COPY_VECTORSTORE_FILE_NAME),
    )

    return cls(
        ad_copy_vectorstore=ad_copy_vectorstore,
        persist_path=persist_path,
        **params,
    )

  def write(self) -> None:
    """Writes the model to the persist path specified in the constructor."""
    init_fields = [
        field.name for field in dataclasses.fields(self) if field.init
    ]
    not_param = {"ad_copy_vectorstore", "persist_path"}
    params = {
        field: getattr(self, field)
        for field in init_fields
        if field not in not_param
    }
    params["ad_format_params"] = params.pop("ad_format").dict()
    params["unique_headlines"] = list(params["unique_headlines"])
    params["unique_descriptions"] = list(params["unique_descriptions"])

    path = pathlib.Path(self.persist_path)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / PARAMS_FILE_NAME, "w") as f:
      json.dump(params, f)

    self.ad_copy_vectorstore.write()

  def construct_response(
      self,
      google_ad: GoogleAd,
      keywords: str,
      headlines_are_memorised: bool,
      descriptions_are_memorised: bool,
      error_message: str,
  ) -> CopycatResponse:
    """Constructs a CopycatResponse from a generated GoogleAd.

    Args:
      google_ad: The generated ad object.
      keywords: The keywords used to generate the ad.
      headlines_are_memorised: Whether the headlines are memorised.
      descriptions_are_memorised: Whether the descriptions are memorised.
      error_message: The error message if the generation was not successful.

    Returns:
      A CopycatResponse object.
    """
    response = CopycatResponse(
        google_ad=google_ad,
        keywords=keywords,
        headlines_are_memorised=headlines_are_memorised,
        descriptions_are_memorised=descriptions_are_memorised,
        error_message=error_message,
    )
    if not self.ad_copy_evaluator.is_underpopulated(response.google_ad):
      response.evaluation_metrics = ad_copy_evaluator.evaluate_ad_copy(
          google_ad=google_ad,
          keywords=keywords,
          ad_copy_vectorstore=self.ad_copy_vectorstore,
      )
    return response

  def construct_text_generation_request_for_new_ad_copy(
      self,
      *,
      keywords: str,
      keywords_specific_instructions: str = "",
      style_guide: str = "",
      system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
      num_in_context_examples: int = 10,
      in_context_examples_fetch_k: int = 1000,
      in_context_examples_lambda_mult: float = 0.5,
      model_name: ModelName | str = ModelName.GEMINI_1_5_FLASH,
      temperature: float = 0.95,
      top_k: int = 20,
      top_p: float = 0.95,
      safety_settings: ad_copy_generator.SafetySettingsType | None = None,
      system_instruction_kwargs: dict[str, Any] | None = None,
  ) -> TextGenerationRequest:
    """Constructs a request for generating a new ad copy.

    This prompt consists of a system prompt, a style guide, and a number of
    in context examples. The in context examples are retrieved from the ad copy
    vectorstore.

    Args:
      keywords: The keywords to use to generate the ad copy.
      keywords_specific_instructions: The keywords specific instructions to use.
      style_guide: The style guide to use.
      system_instruction: The system instruction to use.
      num_in_context_examples: The number of in context examples to use.
      in_context_examples_fetch_k: The number of ads to fetch for in context
        examples.
      in_context_examples_lambda_mult: The lambda multiplier to use for in
        context examples.
      model_name: The name of the gemini model to use.
      temperature: The temperature to use for the chat model.
      top_k: The top-k to use for the chat model.
      top_p: The top-p to use for the chat model.
      safety_settings: The safety settings for the chat model.
      system_instruction_kwargs: Additional arguments to pass to the system
        instruction.

    Returns:
      A text generation request, containing the prompt, system instruction, and
      model parameters.
    """
    system_instruction_kwargs = system_instruction_kwargs or {}
    default_system_instruction_kwargs = {
        "max_headlines": self.ad_format.max_headlines,
        "max_descriptions": self.ad_format.max_descriptions,
    }
    system_instruction_kwargs = (
        default_system_instruction_kwargs | system_instruction_kwargs
    )
    system_instruction = (
        ad_copy_generator.construct_system_instruction(
            system_instruction=system_instruction,
            style_guide=style_guide,
            system_instruction_kwargs=system_instruction_kwargs,
        )
    )

    relavent_example_ads = self.ad_copy_vectorstore.get_relevant_ads(
        keywords,
        k=num_in_context_examples,
        fetch_k=in_context_examples_fetch_k,
        lambda_mult=in_context_examples_lambda_mult,
    )
    in_context_example_content = (
        ad_copy_generator.construct_examples_for_new_ad_copy_generation(
            relavent_example_ads
        )
    )

    prompt = ad_copy_generator.construct_new_ad_copy_prompt(
        in_context_example_content=in_context_example_content,
        keywords=keywords,
        keywords_specific_instructions=keywords_specific_instructions,
    )

    return TextGenerationRequest(
        prompt=prompt,
        system_instruction=system_instruction,
        chat_model_name=ModelName(model_name),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        safety_settings=safety_settings,
    )

  def generate_new_ad_copy(
      self,
      *,
      keywords: list[str],
      keywords_specific_instructions: list[str] | None = None,
      style_guide: str = "",
      system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
      num_in_context_examples: int = 10,
      in_context_examples_fetch_k: int = 1000,
      in_context_examples_lambda_mult: float = 0.5,
      model_name: ModelName | str = ModelName.GEMINI_1_5_FLASH,
      temperature: float = 0.95,
      top_k: int = 20,
      top_p: float = 0.95,
      allow_memorised_headlines: bool = True,
      allow_memorised_descriptions: bool = False,
      safety_settings: ad_copy_generator.SafetySettingsType | None = None,
      system_instruction_kwargs: dict[str, Any] | None = None,
  ) -> list[CopycatResponse]:
    """Generates a new ad copy.

    Args:
      keywords: The keywords to use to generate the ad copy.
      keywords_specific_instructions: The keywords specific instructions to use.
        Defaults to a list of empty strings.
      style_guide: The style guide to use.
      system_instruction: The system instruction to use.
      num_in_context_examples: The number of in context examples to use.
      in_context_examples_fetch_k: The number of ads to fetch for in context
        examples.
      in_context_examples_lambda_mult: The lambda multiplier to use for in
        context examples.
      model_name: The name of the chat model to use.
      temperature: The temperature to use for the chat model.
      top_k: The top-k to use for the chat model.
      top_p: The top-p to use for the chat model.
      allow_memorised_headlines: Whether to allow memorised headlines.
      allow_memorised_descriptions: Whether to allow memorised descriptions.
      safety_settings: The safety settings for the chat model.
      system_instruction_kwargs: Additional arguments to pass to the system
        instruction.

    Returns:
      A CopycatResponse object.

    Raises:
      ValueError: If keywords and keywords_specific_instructions have different
        lengths.
      RuntimeError: If the number of responses does not match the number of
        keywords. This shouldn't happen, if it happens it indicates a bug in the
        code.
    """
    empty_ad_copy = GoogleAd(headlines=[], descriptions=[])

    if keywords_specific_instructions is None:
      keywords_specific_instructions = [""] * len(keywords)

    if len(keywords) != len(keywords_specific_instructions):
      raise ValueError(
          "keywords and keywords_specific_instructions must have the same"
          " length."
      )

    params = dict(
        num_in_context_examples=num_in_context_examples,
        in_context_examples_fetch_k=in_context_examples_fetch_k,
        in_context_examples_lambda_mult=in_context_examples_lambda_mult,
        style_guide=style_guide,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        safety_settings=safety_settings,
        system_instruction_kwargs=system_instruction_kwargs,
    )
    requests = [
        self.construct_text_generation_request_for_new_ad_copy(
            keywords=keywords_i,
            keywords_specific_instructions=keywords_specific_instructions_i,
            **params,
        )
        for keywords_i, keywords_specific_instructions_i in zip(
            keywords, keywords_specific_instructions
        )
    ]

    candidates = [
        response.candidates[0]
        for response in ad_copy_generator.generate_google_ad_json_batch(
            requests
        )
    ]

    responses = []
    for keywords_i, candidate_i in zip(keywords, candidates):
      if candidate_i.finish_reason is not ad_copy_generator.FinishReason.STOP:
        responses.append(
            self.construct_response(
                google_ad=empty_ad_copy,
                keywords=keywords_i,
                headlines_are_memorised=None,
                descriptions_are_memorised=None,
                error_message=f"- {candidate_i}",
            )
        )
        continue

      try:
        ad_copy = GoogleAd.model_validate_json(
            candidate_i.content.parts[0].text
        )
      except ValidationError as e:
        responses.append(
            self.construct_response(
                google_ad=empty_ad_copy,
                keywords=keywords_i,
                headlines_are_memorised=None,
                descriptions_are_memorised=None,
                error_message=f"- {e}",
            )
        )
        continue

      ad_copy_generator.remove_invalid_headlines_and_descriptions(
          ad_copy, self.ad_format
      )

      evaluation_results = self.ad_copy_evaluator.evaluate(
          ad_copy,
          allow_memorised_headlines=allow_memorised_headlines,
          allow_memorised_descriptions=allow_memorised_descriptions,
      )

      responses.append(
          self.construct_response(
              google_ad=ad_copy,
              keywords=keywords_i,
              headlines_are_memorised=evaluation_results.headlines_are_memorised,
              descriptions_are_memorised=evaluation_results.descriptions_are_memorised,
              error_message="\n".join(
                  map(lambda x: f"- {x}", sorted(evaluation_results.errors))
              ),
          )
      )

    if len(responses) != len(keywords):
      raise RuntimeError(
          "The number of responses does not match the number of keywords."
      )

    return responses
