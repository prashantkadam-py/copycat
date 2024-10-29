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

"""A collection of re-usable event handlers for the Copycat UI."""

import dataclasses
import json
import logging

import vertexai
import mesop as me
import numpy as np
import pandas as pd

from copycat import copycat
from copycat.data import sheets
from copycat.data import utils as data_utils
from copycat.ui import states


def update_copycat_parameter(event: me.InputEvent) -> None:
  """Updates a parameter in the CopycatParamsState.

  Args:
    event: The input event to handle. This can be any event where the key is
      set.

  Raises:
    ValueError: If the key is not a field in CopycatParamsState.
  """
  params = me.state(states.CopycatParamsState)
  for field in dataclasses.fields(params):
    if field.name == event.key:
      setattr(params, event.key, field.type(event.value))
      return
  raise ValueError(f"Field {event.key} does not exist in CopycatParamsState.")


def update_app_state_parameter(event: me.InputEvent) -> None:
  """Updates a parameter in the AppState.

  Args:
    event: The input event to handle. This can be any event where the key is
      set.

  Raises:
    ValueError: If the key is not a field in AppState.
  """
  state = me.state(states.AppState)
  for field in dataclasses.fields(state):
    if field.name == event.key:
      setattr(state, event.key, field.type(event.value))
      return
  raise ValueError(f"Field {event.key} does not exist in AppState.")


def update_copycat_parameter_from_slide_toggle(
    event: me.SlideToggleChangeEvent,
) -> None:
  """Updates a copycat parameter from a slide toggle change event.

  Args:
    event: The slide toggle change event to handle.
  """
  state = me.state(states.CopycatParamsState)
  setattr(state, event.key, not getattr(state, event.key))


def language_on_blur(event: me.InputBlurEvent) -> None:
  """Updates the language and the embedding model name based on the language.

  Args:
    event: The input blur event to handle.
  """
  state = me.state(states.CopycatParamsState)
  state.language = event.value

  if "english" in event.value.lower():
    state.embedding_model_name = copycat.EmbeddingModelName.TEXT_EMBEDDING.value
  else:
    state.embedding_model_name = (
        copycat.EmbeddingModelName.TEXT_MULTILINGUAL_EMBEDDING.value
    )

  send_log(
      f"Updating embedding model name to {state.embedding_model_name} for"
      f" language = {state.language}"
  )


def ad_format_on_change(event: me.RadioChangeEvent) -> None:
  """Updates the ad format and related parameters in the CopycatParamsState.

  Args:
    event: The radio change event to handle.
  """
  state = me.state(states.CopycatParamsState)
  state.ad_format = event.value

  if state.ad_format != "custom":
    send_log(
        f"Updating max headlines and descriptions for {state.ad_format} ad"
        " format"
    )
    ad_format = copycat.google_ads.get_google_ad_format(event.value)
    state.max_headlines = ad_format.max_headlines
    state.max_descriptions = ad_format.max_descriptions
  else:
    send_log(
        f"Max headlines and descriptions not updated for {state.ad_format} ad"
        " format"
    )


def embedding_model_dimensionality_on_blur(event: me.InputBlurEvent) -> None:
  """Updates the embedding model dimensionality based on the input value.

  The value is clamped to the range [10, 768].

  Args:
    event: The input blur event to handle.
  """
  state = me.state(states.CopycatParamsState)
  raw_value = int(event.value)
  if raw_value > 786:
    state.embedding_model_dimensionality = 768
  elif raw_value < 10:
    state.embedding_model_dimensionality = 10
  else:
    state.embedding_model_dimensionality = raw_value


def close_starting_dialog(event: me.ClickEvent) -> None:
  """Closes the starting dialog.

  This clears the new Google Sheet URL and name, and sets the
  show_starting_dialog state to False.

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  state.new_google_sheet_url = ""
  state.new_google_sheet_name = ""
  state.show_starting_dialog = False


def open_starting_dialog(event: me.ClickEvent) -> None:
  """Opens the starting dialog by setting show_starting_dialog to True.

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  state.show_starting_dialog = True


def reset_state(
    state: type[states.AppState] | type[states.CopycatParamsState],
) -> None:
  """Resets a state to its default values.

  Args:
    state: The state to reset.
  """
  send_log(f"Resetting state: {state}")
  params = me.state(state)

  for field in dataclasses.fields(params):
    if field.default is not dataclasses.MISSING:
      setattr(params, field.name, field.default)
    elif field.default_factory is not dataclasses.MISSING:
      setattr(params, field.name, field.default_factory())
    else:
      setattr(params, field.name, field.type())


def save_params_to_google_sheet(event: me.ClickEvent) -> None:
  """Saves the Copycat parameters to the Google Sheet.

  The parameters are written to a tab named "READ ONLY: Copycat Params".

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  params = me.state(states.CopycatParamsState)

  params_table = pd.DataFrame([dataclasses.asdict(params)])
  params_table["Parameter Name"] = "Parameter Value"
  params_table = params_table.set_index("Parameter Name")

  sheet = sheets.GoogleSheet.load(state.google_sheet_url)
  sheet["READ ONLY: Copycat Params"] = params_table

  send_log("Copycat params saved to sheet")


def load_params_from_google_sheet(event: me.ClickEvent) -> None:
  """Loads the Copycat parameters from the Google Sheet.

  The parameters are read from a tab named "READ ONLY: Copycat Params".

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  params = me.state(states.CopycatParamsState)

  sheet = sheets.GoogleSheet.load(state.google_sheet_url)
  params_table = sheet["READ ONLY: Copycat Params"]

  for field in dataclasses.fields(params):
    if field.name in params_table:
      param_value = params_table[field.name].values[0]
      if field.type is bool and param_value == "TRUE":
        param_value = True
      elif field.type is bool and param_value == "FALSE":
        param_value = False
      else:
        param_value = field.type(param_value)
      setattr(params, field.name, param_value)

  state.has_copycat_instance = "READ ONLY: Copycat Instance Params" in sheet
  send_log("Loaded Copycat params from sheet")


def create_new_google_sheet(event: me.ClickEvent) -> None:
  """Creates a new Google Sheet and initializes it with the default tabs.

  The default tabs are:
    - Training Ads
    - New Keywords
    - Extra Instructions for New Ads

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  sheet = sheets.GoogleSheet.new(state.new_google_sheet_name)
  start_logger(sheet.url)

  reset_state(states.AppState)
  reset_state(states.CopycatParamsState)
  state = me.state(states.AppState)

  state.google_sheet_url = sheet.url
  state.google_sheet_name = sheet.title

  sheet["Training Ads"] = pd.DataFrame(
      columns=[
          "Campaign ID",
          "Ad Group",
          "URL",
          "Ad Strength",
          "Keywords",
      ]
      + [f"Headline {i}" for i in range(1, 16)]
      + [f"Description {i}" for i in range(1, 5)],
  ).set_index(["Campaign ID", "Ad Group"])
  sheet["New Keywords"] = pd.DataFrame(
      columns=["Campaign ID", "Ad Group", "Keyword"],
  ).set_index(["Campaign ID", "Ad Group"])
  sheet["Extra Instructions for New Ads"] = pd.DataFrame(
      columns=["Campaign ID", "Ad Group", "Extra Instructions"],
  ).set_index(["Campaign ID", "Ad Group"])

  sheet.delete_worksheet("Sheet1")
  save_params_to_google_sheet(event)
  close_starting_dialog(event)
  send_log("New Google Sheet created")


def load_existing_google_sheet(event: me.ClickEvent) -> None:
  """Loads an existing Google Sheet.

  The sheet should contain the following tabs:
    - Training Ads
    - New Keywords
    - Extra Instructions for New Ads

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  sheet = sheets.GoogleSheet.load(state.new_google_sheet_url)
  start_logger(sheet.url)

  reset_state(states.AppState)
  reset_state(states.CopycatParamsState)
  state = me.state(states.AppState)

  # Load sheet
  state.google_sheet_url = sheet.url
  state.google_sheet_name = sheet.title

  if "READ ONLY: Copycat Params" in sheet:
    load_params_from_google_sheet(event)
  else:
    save_params_to_google_sheet(event)

  close_starting_dialog(event)
  send_log("Existing Google Sheet loaded")


def start_logger(url: str) -> None:
  """Starts the logger and writes logs to a Google Sheet.

  Args:
    url: The URL of the Google Sheet to write logs to.
  """
  handler = sheets.GoogleSheetsHandler(sheet_url=url, log_worksheet_name="Logs")
  handler.setLevel(logging.INFO)
  logger = logging.getLogger("copycat")
  logger.handlers = []
  logger.addHandler(handler)
  logger.setLevel(logging.INFO)
  logger.info("Logger Started")


def send_log(message: str, level: int = logging.INFO) -> None:
  """Sends a log message to the logger.

  Args:
    message: The log message to send.
    level: The level of the log message. Defaults to INFO.
  """
  logger = logging.getLogger("copycat.ui")
  logger.log(level=level, msg=message)


def update_log_level(event: me.SelectSelectionChangeEvent) -> None:
  """Updates the log level of the logger.

  Args:
    event: The select selection change event to handle.
  """
  state = me.state(states.AppState)
  state.log_level = int(event.value)
  logger = logging.getLogger("copycat")
  logger.handlers[0].setLevel(state.log_level)


def show_hide_google_sheet(event: me.ClickEvent) -> None:
  """Shows or hides the Google Sheet preview panel.

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  state.display_google_sheet = not state.display_google_sheet


def validate_sheet(event: me.ClickEvent) -> None:
  """Validates the Google Sheet.

  The sheet is validated by checking that it contains the required tabs,
  index columns, and columns, and that it has the minimum number of rows.

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  sheet_url = state.google_sheet_url
  send_log(f"Validating {sheet_url}")

  sheet = sheets.GoogleSheet.load(sheet_url)
  send_log(f"Sheet Name = {sheet.title}")

  # Validate all required sheets exist, have the correct index and columns,
  # and have the minimum number of rows.
  required_index_names = ["Campaign ID", "Ad Group"]
  required_columns = {
      "Training Ads": set([
          "URL",
          "Ad Strength",
          "Keywords",
          "Headline 1",
          "Description 1",
      ]),
      "New Keywords": set([
          "Keyword",
      ]),
      "Extra Instructions for New Ads": set([
          "Extra Instructions",
      ]),
  }
  min_rows = {
      "Training Ads": 1,
      "New Keywords": 1,
      "Extra Instructions for New Ads": 0,
  }

  state.google_sheet_is_valid = True
  for sheet_name in [
      "Training Ads",
      "New Keywords",
      "Extra Instructions for New Ads",
  ]:
    if sheet_name in sheet:
      send_log(f"{sheet_name} sheet found")
    else:
      send_log(
          f"VALIDATION FAILED: {sheet_name} sheet not found.", logging.ERROR
      )
      state.google_sheet_is_valid = False

    worksheet = sheet[sheet_name]
    actual_index_names = list(worksheet.index.names)
    if required_index_names != actual_index_names:
      send_log(
          f"VALIDATION FAILED: {sheet_name} requires index columns:"
          f" {required_index_names}, but found {actual_index_names}.",
          logging.ERROR,
      )
      state.google_sheet_is_valid = False

    actual_columns = set(worksheet.columns.values.tolist())
    extra_columns = actual_columns - required_columns[sheet_name]
    missing_columns = required_columns[sheet_name] - actual_columns

    if missing_columns:
      send_log(
          f"VALIDATION FAILED: Missing columns in {sheet_name}:"
          f" {missing_columns}",
          logging.ERROR,
      )
      state.google_sheet_is_valid = False
    else:
      send_log(f"All required columns in {sheet_name}")

    if extra_columns:
      send_log(f"{sheet_name} has the following extra columns: {extra_columns}")

    n_rows = len(worksheet)
    if n_rows < min_rows[sheet_name]:
      send_log(
          f"VALIDATION FAILED: {sheet_name} sheet has fewer than the minimum"
          f" number of rows: min={min_rows[sheet_name]}.",
          logging.ERROR,
      )
      state.google_sheet_is_valid = False
    else:
      send_log(f"{sheet_name} has {n_rows:,} rows")

  # Log the number of headline and description columns in the training ads
  training_ads = sheet["Training Ads"]
  n_headline_columns = len(
      [c for c in training_ads.columns if c.startswith("Headline")]
  )
  n_description_columns = len(
      [c for c in training_ads.columns if c.startswith("Description")]
  )
  send_log(f"Training Ads have up to {n_headline_columns} headlines.")
  send_log(f"Training Ads have up to {n_description_columns} descriptions.")

  # Completed validation
  if state.google_sheet_is_valid:
    send_log("VALIDATION COMPLETED: Google Sheet is valid")
  else:
    send_log("VALIDATION COMPLETED: Google Sheet is invalid", logging.ERROR)


def save_copycat_to_sheet(
    sheet: sheets.GoogleSheet, model: copycat.Copycat
) -> None:
  """Saves the Copycat model to the Google Sheet.

  The model is saved in two tabs:
    - READ ONLY: Training Ad Exemplars: Contains the exemplar ads.
    - READ ONLY: Copycat Instance Params: Contains the other model parameters.

  Args:
    sheet: The Google Sheet to save the model to.
    model: The Copycat model to save.
  """
  send_log("Saving Copycat instance to sheet")
  model_params = model.to_dict()

  # Store the exemplar ads in their own sheet
  exemplars_dict = model_params["ad_copy_vectorstore"].pop("ad_exemplars")
  ad_exemplars = pd.DataFrame.from_dict(exemplars_dict, orient="tight")
  ad_exemplars["embeddings"] = ad_exemplars["embeddings"].apply(
      lambda x: ", ".join(list(map(str, x)))
  )
  ad_exemplars = data_utils.explode_headlines_and_descriptions(ad_exemplars)
  ad_exemplars.index.name = "Exemplar Number"
  sheet["READ ONLY: Training Ad Exemplars"] = ad_exemplars

  # Store the other params as a json string
  other_params = pd.DataFrame([{
      "params_json": json.dumps(model_params),
  }])
  sheet["READ ONLY: Copycat Instance Params"] = other_params


def load_copycat_from_sheet(sheet: sheets.GoogleSheet) -> copycat.Copycat:
  """Loads a Copycat instance from the Google Sheet.

  The instance is loaded from two tabs:
    - READ ONLY: Training Ad Exemplars: Contains the exemplar ads.
    - READ ONLY: Copycat Instance Params: Contains the other model parameters.

  Args:
    sheet: The Google Sheet to load the instance from.

  Returns:
    The Copycat instance.
  """
  send_log("Loading Copycat instance from sheet")
  instance_json = sheet["READ ONLY: Copycat Instance Params"].loc[
      0, "params_json"
  ]
  instance_dict = json.loads(instance_json)

  ad_exemplars = sheet["READ ONLY: Training Ad Exemplars"]
  ad_exemplars["embeddings"] = ad_exemplars["embeddings"].apply(
      lambda x: list(map(float, x.split(", ")))
  )
  ad_exemplars = data_utils.collapse_headlines_and_descriptions(ad_exemplars)
  ad_exemplars_dict = ad_exemplars.to_dict(orient="tight")

  instance_dict["ad_copy_vectorstore"]["ad_exemplars"] = ad_exemplars_dict
  copycat_instance = copycat.Copycat.from_dict(instance_dict)
  return copycat_instance


def build_new_copycat_instance(event: me.ClickEvent):
  """Builds a new Copycat instance from the Google Sheet.

  The Copycat instance is created using the parameters from the Google Sheet
  and the CopycatParamsState. The instance is then saved to the Google Sheet.

  Args:
    event: The click event to handle.
  """
  state = me.state(states.AppState)
  params = me.state(states.CopycatParamsState)
  sheet = sheets.GoogleSheet.load(state.google_sheet_url)
  save_params_to_google_sheet(event)

  vertexai.init(
      project=params.vertex_ai_project_id, location=params.vertex_ai_location
  )

  train_data = data_utils.collapse_headlines_and_descriptions(
      sheet["Training Ads"]
  )
  train_data = train_data.rename({"Keywords": "keywords"}, axis=1)
  train_data = train_data[["headlines", "descriptions", "keywords"]]
  train_data = train_data.loc[train_data["headlines"].apply(len) > 0]

  send_log(f"Loaded {len(train_data)} rows of raw data from the Google Sheet.")

  if params.ad_format == "custom":
    ad_format = copycat.google_ads.GoogleAdFormat(
        name="custom",
        max_headlines=params.max_headlines,
        max_descriptions=params.max_descriptions,
        min_headlines=1,
        min_descriptions=1,
        max_headline_length=30,
        max_description_length=90,
    )
    send_log("Using a custom ad format.")
  else:
    ad_format = copycat.google_ads.get_google_ad_format(params.ad_format)
    send_log(f"Using the following ad format: {ad_format.name}")

  affinity_preference = (
      params.custom_affinity_preference
      if params.use_custom_affinity_preference
      else None
  )
  send_log(
      "Affinity preference:"
      f" {affinity_preference} (custom={params.use_custom_affinity_preference})"
  )

  send_log("Creating Copycat.")

  model = copycat.Copycat.create_from_pandas(
      training_data=train_data,
      ad_format=ad_format,
      on_invalid_ad=params.on_invalid_ad,
      embedding_model_name=params.embedding_model_name,
      embedding_model_dimensionality=params.embedding_model_dimensionality,
      embedding_model_batch_size=params.embedding_model_batch_size,
      vectorstore_exemplar_selection_method=params.exemplar_selection_method,
      vectorstore_max_initial_ads=params.max_initial_ads,
      vectorstore_max_exemplar_ads=params.max_exemplar_ads,
      vectorstore_affinity_preference=affinity_preference,
      replace_special_variables_with_default=params.how_to_handle_special_variables
      == "replace",
  )
  send_log(
      "Copycat instance created with"
      f" {model.ad_copy_vectorstore.n_exemplars} exemplar ads."
  )

  save_copycat_to_sheet(sheet, model)
  state.has_copycat_instance = True

  send_log("Copycat instance stored in google sheet.")


def generate_style_guide(event: me.ClickEvent):
  """Generates a style guide from the Google Sheet.

  The style guide is generated using the parameters from the Google Sheet
  and the CopycatParamsState. The style guide is then saved to the Google Sheet.

  Args:
    event: The click event to handle.
  """
  send_log("Generating style guide")
  state = me.state(states.AppState)
  params = me.state(states.CopycatParamsState)
  sheet = sheets.GoogleSheet.load(state.google_sheet_url)

  vertexai.init(
      project=params.vertex_ai_project_id,
      location=params.vertex_ai_location,
  )

  copycat_instance = load_copycat_from_sheet(sheet)

  send_log("Preparing to generate style guide")
  style_guide_generator = copycat.StyleGuideGenerator()
  if params.style_guide_files_uri:
    send_log(
        f"Checking for files in the GCP bucket {params.style_guide_files_uri}"
    )
    style_guide_generator.get_all_files(params.style_guide_files_uri)

  send_log("Generating style guide")
  model_response = style_guide_generator.generate_style_guide(
      brand_name=params.company_name,
      ad_copy_vectorstore=copycat_instance.ad_copy_vectorstore,
      additional_style_instructions=params.style_guide_additional_instructions,
      model_name=params.style_guide_chat_model_name,
      safety_settings=copycat.ALL_SAFETY_SETTINGS_ONLY_HIGH,
      temperature=params.style_guide_temperature,
      top_k=params.style_guide_top_k,
      top_p=params.style_guide_top_p,
  )

  params.style_guide = model_response.candidates[0].content.text
  send_log("Style guide generated")

  save_params_to_google_sheet(event)
