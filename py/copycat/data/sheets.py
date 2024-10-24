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

"""Utility functions for loading and saving data to Google Sheets."""

import logging
import time
from typing import Any
import google.auth.credentials
import gspread
import pandas as pd


HEADING_FORMAT = {
    "backgroundColor": {"red": 0.3, "green": 0.3, "blue": 0.3},
    "horizontalAlignment": "CENTER",
    "textFormat": {
        "bold": True,
        "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
    },
}


GOOGLE_AUTH_CREDENTIALS: google.auth.credentials.Credentials | None = None


def set_google_auth_credentials(
    credentials: google.auth.credentials.Credentials,
) -> None:
  """Sets the Google auth credentials.

  These credentials are used to authorize the Google Sheets client. You can get
  the credentials by calling google.auth.default().
  """
  global GOOGLE_AUTH_CREDENTIALS
  GOOGLE_AUTH_CREDENTIALS = credentials


def get_gspread_client() -> gspread.Client:
  """Creates a Google Sheets client."""
  if GOOGLE_AUTH_CREDENTIALS is None:
    raise ValueError(
        "Google auth credentials are not set. First call "
        "set_google_auth_credentials()"
    )
  return gspread.authorize(GOOGLE_AUTH_CREDENTIALS)


class GoogleSheet:
  """An object for reading and writing data to Google Sheets.

  This class provides a convenient way to access and manipulate data in Google
  Sheets. It allows you to read and write pandas dataframes to/from Google
  Sheets.

  Attributes:
    spreadsheet: The Google Sheets spreadsheet.
    url: The url of the spreadsheet.
    title: The title of the spreadsheet.
  """

  def __init__(self, spreadsheet: gspread.Spreadsheet):
    """Initializes the GoogleSheet object.

    Args:
      spreadsheet: The Google Sheets spreadsheet.
    """
    self.spreadsheet = spreadsheet

  @property
  def url(self) -> str:
    return self.spreadsheet.url

  @property
  def title(self) -> str:
    return self.spreadsheet.title

  def __str__(self):
    return "\n".join([
        "GoogleSheet:",
        f"  URL: {self.url}",
        f"  Name: {self.title}",
        (
            "  Worksheet Names:"
            f" {[worksheet.title for worksheet in self.spreadsheet.worksheets()]}"
        ),
    ])

  @classmethod
  def new(cls, spreadsheet_name: str) -> "GoogleSheet":
    """Creates a Google Sheet DataFrame with a new spreadsheet.

    Args:
      spreadsheet_name: The name of the spreadsheet to create.

    Returns:
      A GoogleSheet object referencing the new spreadsheet.
    """
    spreadsheet = get_gspread_client().create(spreadsheet_name)
    return cls(spreadsheet)

  @classmethod
  def load(cls, url: str) -> "GoogleSheet":
    """Loads a Google Sheet DataFrame from a URL.

    Args:
      url: The URL of the spreadsheet to load.

    Returns:
      A GoogleSheet object referencing the spreadsheet.
    """
    spreadsheet = get_gspread_client().open_by_url(url)
    return cls(spreadsheet)

  def __contains__(self, worksheet_title: str) -> bool:
    """Checks if a worksheet with the given title exists in the spreadsheet.

    Args:
      worksheet_title: The title of the worksheet to check.

    Returns:
      True if the worksheet exists, False otherwise.
    """
    for worksheet in self.spreadsheet.worksheets():
      if worksheet.title == worksheet_title:
        return True
    return False

  def __getitem__(self, worksheet_title: str) -> pd.DataFrame:
    """Gets the data from a worksheet with the given title.

    The column headings are assumed to be the first row of the worksheet.
    The row index is assumed to be all of the frozen columns in the worksheet.

    Args:
      worksheet_title: The title of the worksheet to get data from.

    Returns:
      A pandas DataFrame containing the data from the worksheet.
    """
    worksheet = self.spreadsheet.worksheet(worksheet_title)
    data = pd.DataFrame(
        worksheet.get_all_records(value_render_option="UNFORMATTED_VALUE")
    )

    index_cols = data.columns.values[: worksheet.frozen_col_count].tolist()
    if index_cols:
      data = data.set_index(index_cols)

    return data

  def _parse_data(
      self, data: pd.DataFrame
  ) -> tuple[list[list[Any]], list[str], int]:
    """Parses the data into a format suitable for writing to Google Sheets.

    Args:
      data: A pandas DataFrame containing the data to parse.

    Returns:
      A tuple containing the data to write, the column names, and the number of
      index columns.
    """
    n_index_cols = data.index.nlevels
    data_reset = data.reset_index()
    column_names = data_reset.columns.tolist()
    row_values = data_reset.values.tolist()

    if not row_values:
      # If there is no data, then just have a single empty row
      row_values = [[""] * len(column_names)]

    data_to_write = [column_names] + row_values
    return data_to_write, column_names, n_index_cols

  def _overwrite_worksheet(
      self, worksheet_title: str, data: list[list[Any]], n_index_cols: int
  ) -> None:
    """Overwrites the data in a worksheet with the given title.

    If the worksheet does not exist, then it is created.

    Args:
      worksheet_title: The title of the worksheet to overwrite.
      data: A list of lists containing the data to write.
      n_index_cols: The number of columns to freeze as indexes.
    """
    if worksheet_title in self:
      worksheet = self.spreadsheet.worksheet(worksheet_title)
      worksheet.clear()
    else:
      worksheet = self.spreadsheet.add_worksheet(
          title=worksheet_title,
          rows=len(data),
          cols=len(data[0]),
      )
    self._update_size_of_worksheet(worksheet_title, data)
    worksheet.update(data)
    self._update_worksheet_formatting(worksheet_title, n_index_cols)

  def _update_worksheet_formatting(
      self, worksheet_title: str, n_index_cols: int
  ) -> None:
    """Updates the formatting of the worksheet."""
    worksheet = self.spreadsheet.worksheet(worksheet_title)
    worksheet.freeze(rows=1, cols=n_index_cols)
    worksheet.format("1:1", HEADING_FORMAT)

  def _update_size_of_worksheet(
      self, worksheet_title: str, data: list[list[Any]]
  ) -> None:
    """Updates the size of the worksheet to fit the data."""
    worksheet = self.spreadsheet.worksheet(worksheet_title)
    target_rows = len(data)
    target_cols = len(data[0])
    if worksheet.row_count < target_rows:
      worksheet.add_rows(target_rows - worksheet.row_count)
    elif worksheet.row_count > target_rows:
      worksheet.delete_rows(1, worksheet.row_count - target_rows)

    if worksheet.col_count < target_cols:
      worksheet.add_cols(target_cols - worksheet.col_count)
    elif worksheet.col_count > target_cols:
      worksheet.delete_columns(1, worksheet.col_count - target_cols)

  def _construct_update_batches(
      self,
      data_to_write: list[list[Any]],
      existing_data: list[list[Any]],
  ) -> list[dict[str, Any]]:
    """Constructs update batches for Google Sheets.

    The batches are the rows where the data is different.

    Args:
      data_to_write: A list of lists containing the data to write.
      existing_data: A list of lists containing the existing data.

    Returns:
      A list of update batches.
    """
    update_batches = []
    last_different_row_index = -1
    for row_index, (new_row, old_row) in enumerate(
        zip(data_to_write, existing_data), start=1
    ):
      if new_row != old_row:
        if last_different_row_index == (row_index - 1):
          update_batches[-1]["end_row"] = row_index
          update_batches[-1]["values"].append(new_row)
        else:
          update_batches.append({
              "start_row": row_index,
              "end_row": row_index,
              "values": [new_row],
          })
        last_different_row_index = row_index

    if len(data_to_write) > len(existing_data):
      update_batches.append({
          "start_row": len(existing_data) + 1,
          "end_row": len(data_to_write),
          "values": data_to_write[len(existing_data) :],
      })

    update_batches = [
        {
            "range": (
                gspread.utils.rowcol_to_a1(batch["start_row"], 1)
                + ":"
                + gspread.utils.rowcol_to_a1(
                    batch["end_row"], len(data_to_write[0])
                )
            ),
            "values": batch["values"],
        }
        for batch in update_batches
    ]
    return update_batches

  def __setitem__(self, worksheet_title: str, data: pd.DataFrame) -> None:
    """Sets the data in a worksheet with the given title.

    The first row of the data is set with the column headings.
    The indexes are set as the first column(s) and are frozen.
    If the worksheet already exists and column names are unchanged, then only
    the rows that have changed are updated. If the worksheet already exists and
    the column names are changed, then the entire worksheet is overwritten.

    Args:
      worksheet_title: The title of the worksheet to set data in.
      data: A pandas DataFrame containing the data to set.
    """
    data_to_write, column_names, n_index_cols = self._parse_data(data)

    # If the worksheet does not exist, then just write all the data
    if worksheet_title not in self:
      self._overwrite_worksheet(worksheet_title, data_to_write, n_index_cols)
      return

    existing_data, existing_columns, _ = self._parse_data(self[worksheet_title])

    # If the columns are different, overwrite the entire data
    if existing_columns != column_names:
      self._overwrite_worksheet(worksheet_title, data_to_write, n_index_cols)
      return

    # Otherwise only update rows that have changed
    update_batches = self._construct_update_batches(
        data_to_write, existing_data
    )
    self._update_size_of_worksheet(worksheet_title, data_to_write)
    self.spreadsheet.worksheet(worksheet_title).batch_update(update_batches)
    self._update_worksheet_formatting(worksheet_title, n_index_cols)

  def delete_worksheet(self, worksheet_title: str) -> None:
    """Deletes the worksheet with the given title."""
    worksheet = self.spreadsheet.worksheet(worksheet_title)
    self.spreadsheet.del_worksheet(worksheet)


class GoogleSheetsLogSender:

  HEADINGS = ["UTC Timestamp", "Log Level", "Logger Name", "Message"]

  def __init__(
      self,
      sheet_url: str,
      log_worksheet_name: str = "Logs",
  ):
    self.client = get_gspread_client()
    self.spreadsheet = self.client.open_by_url(sheet_url)

    existing_worksheets = [
        sheet.title for sheet in self.spreadsheet.worksheets()
    ]
    if log_worksheet_name not in existing_worksheets:
      self.log_worksheet = self.spreadsheet.add_worksheet(
          title=log_worksheet_name, rows=2, cols=4
      )
      self.log_worksheet.update(range_name="A1:D1", values=[self.HEADINGS])
      self.log_worksheet.format(ranges=["A1:D1"], format=HEADING_FORMAT)
    else:
      self.log_worksheet = self.spreadsheet.worksheet(log_worksheet_name)

    actual_headings = self.log_worksheet.row_values(1)
    if actual_headings != self.HEADINGS:
      raise ValueError(
          "The first row of the log worksheet must be the expected headings:"
          f" {self.HEADINGS}, but got {actual_headings}."
      )

  def write_log(self, msg: logging.LogRecord) -> None:
    self.log_worksheet.insert_row(
        [
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.created)),
            msg.levelname,
            msg.name,
            msg.getMessage(),
        ],
        index=2,
    )


class GoogleSheetsHandler(logging.Handler):

  def __init__(
      self,
      sheet_url: str,
      log_worksheet_name: str = "Logs",
  ) -> None:
    self.sender = GoogleSheetsLogSender(
        sheet_url=sheet_url,
        log_worksheet_name=log_worksheet_name,
    )
    super().__init__()

  def emit(self, record: logging.LogRecord) -> None:
    self.sender.write_log(record)
