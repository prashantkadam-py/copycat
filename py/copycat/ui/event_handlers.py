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
import logging

import mesop as me
import numpy as np
import pandas as pd

from copycat.data import sheets
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

  for param_name in params_table:
    param_value = params_table[param_name].values[0]
    if isinstance(param_value, np.integer):
      param_value = int(param_value)
    elif isinstance(param_value, np.floating):
      param_value = float(param_value)
    else:
      if param_value == "TRUE":
        param_value = True
      elif param_value == "FALSE":
        param_value = False
      else:
        param_value = str(param_value)

    setattr(params, param_name, param_value)

  state.has_copycat_instance = "READ ONLY: Copycat Instance Params" in sheet
  send_log(f"Loaded Copycat params from sheet")


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
