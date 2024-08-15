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

from google.cloud import storage
from vertexai import generative_models

from copycat import ad_copy_generator

ModelName = ad_copy_generator.ModelName

DEFAULT_STYLE_PROMPT = """
In these files is an ad report for {brand_name}, containing their ads (headlines and descriptions)
that they use on Google Search Ads for the corresponding keywords. Headlines and descriptions are lists, and
Google constructs ads by combining those headlines and descriptions together into ads. Therefore the headlines and descriptions
should be sufficiently varied that Google is able to try lots of different combinations in order to find what works best.

Use the ad report to write a comprehensive style guide for this brand's ad copies that can serve as
instruction for a copywriter to write new ad copies for {brand_name} for new lists of keywords.

Additionally, there could be other files included regarding the brand's style that you should consider in the style guide.

additonal style intructions: {additional_style_instructions}

Ensure that you capure strong phrases, slogans and brand names of {brand_name} in the guide.
\n\n
""".replace("\n", " ")

# file types that are accepted for style guide generation with Gemini
ACCEPTED_FILE_TYPES = ["application/pdf", "text/csv"]
SafetySettingsType = (
    dict[generative_models.HarmCategory, generative_models.HarmBlockThreshold]
    | list[generative_models.SafetySetting]
)


class StyleGuideGenerator:
  """A class designed to generate brand style guides using Gemini.

  This class facilitates the retrieval of files from Google Cloud Storage (GCS),
  construction of prompts incorporating these files, and the generation of style
  guides based on provided style instructions and brand information.
  """

  file_info: list[dict[str, str]] = dataclasses.field(default_factory=list)

  def __init__(self) -> None:
    """Initializes a StyleGuideGenerator instance.

    Sets the initial file_info to an empty list.
    """
    self.file_info = []

  def get_all_files(self, bucket_name: str) -> list[dict[str, str]]:
    """Retrieves all file URIs and their MIME types from a GCS bucket.

    Args:
        bucket_name: The name of the GCS bucket.

    Returns:
        A list of dictionaries, each containing information about a file.
        Each dictionary has keys: 'uri' (the GCS URI of the file) and
        'mime_type'.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    file_info = []
    for blob in blobs:
      if blob.content_type in ACCEPTED_FILE_TYPES:
        file_data = {
            "uri": f"gs://{bucket_name}/{blob.name}",
            "mime_type": blob.content_type,
        }
        file_info.append(file_data)
    self.file_info = file_info
    print(f"Successfully retrieved {len(file_info)} file URI's")
    return file_info

  def generate_style_guide(
      self,
      brand_name: str = "",
      additional_style_instructions: str = "",
      model_name: str | ModelName = ModelName.GEMINI_1_5_PRO,
      safety_settings: SafetySettingsType | None = None,
      temperature: float = 0.95,
      top_k: int = 20,
      top_p: float = 0.95,
  ) -> generative_models._generative_models.GenerationResponse:
    """Generates a style guide using the provided model and contents.

    Args:
        brand_name: The name of the brand.
        additional_style_instructions: additional style instructions.
        model_name: The name of the generative model to use.
        safety_settings: The safety settings to use for the model.
        temperature: The temperature parameter for the model (controls
          randomness).
        top_k: The top-k sampling parameter for the model.
        top_p: The top-p sampling parameter for the model.

    Returns:
        The generated style guide text.
    """
    safety_settings = safety_settings or {}

    model_name = ModelName(model_name)
    generation_config = generative_models.GenerationConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    content = self._construct_style_prompt(
        style_prompt_params={
            "brand_name": brand_name,
            "additional_style_instructions": additional_style_instructions,
        }
    )
    model = generative_models.GenerativeModel(
        model_name=model_name.value, safety_settings=safety_settings
    )
    response = model.generate_content(
        contents=[content], generation_config=generation_config
    )
    if not isinstance(response, generative_models.GenerationResponse):
      raise RuntimeError(
          "Response is not a GenerationResponse. Instead got:"
          f" {response}"
      )
    return response

  def _construct_style_prompt(
      self,
      style_prompt: str = DEFAULT_STYLE_PROMPT,
      style_prompt_params: dict[str, str] | None = None,
  ) -> generative_models.Content:
    """Constructs the prompt for style guide generation.

    Args:
        style_prompt: The default style prompt.
        style_prompt_params: The parameters to use for the style prompt.

    Returns:
        The constructed prompt in Content format.
    """
    style_prompt_params = style_prompt_params or {}
    full_style_prompt = style_prompt.format(**style_prompt_params)

    contents: list[generative_models.Part] = [
        generative_models.Part.from_uri(
            data_file["uri"], mime_type=data_file["mime_type"]
        )
        for data_file in self.file_info
        if data_file["mime_type"] in ACCEPTED_FILE_TYPES
    ]

    text_part = generative_models.Part.from_text(full_style_prompt)
    contents.append(text_part)

    if len(contents) == 1:
      print(
          "No files were staged, continuing without brand style files or"
          " examples."
      )

    return generative_models.Content(parts=contents, role="user")
