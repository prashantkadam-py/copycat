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

"""Tests for the sheets module."""

from absl.testing import absltest
from absl.testing import parameterized
import gspread
import pandas as pd

from copycat.data import mock_gspread
from copycat.data import sheets


class GoogleSheetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.gspread_patcher = mock_gspread.PatchGspread()
    self.gspread_patcher.start()

    dummy_credentials = "dummy_credentials"
    sheets.set_google_auth_credentials(dummy_credentials)
    self.client = gspread.authorize(dummy_credentials)

  def tearDown(self):
    super().tearDown()
    self.gspread_patcher.stop()

  def test_can_instantiate_with_new_sheet(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")

    expected_str = "\n".join([
        "GoogleSheet:",
        "  URL: https://mock.sheets.com/spreadsheet/test_sheet",
        "  Name: test_sheet",
        "  Worksheet Names: ['Sheet1']",
    ])
    self.assertEqual(str(google_sheet), expected_str)

  def test_can_instantiate_with_existing_sheet(self):
    spreadsheet = self.client.create("test_sheet")
    spreadsheet.add_worksheet("Another Sheet")

    google_sheet = sheets.GoogleSheet.load(spreadsheet.url)

    expected_str = "\n".join([
        "GoogleSheet:",
        "  URL: https://mock.sheets.com/spreadsheet/test_sheet",
        "  Name: test_sheet",
        "  Worksheet Names: ['Sheet1', 'Another Sheet']",
    ])
    self.assertEqual(str(google_sheet), expected_str)

  def test_can_check_if_sheet_exists(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")

    self.assertIn("Sheet1", google_sheet)
    self.assertNotIn("Another Sheet", google_sheet)

  def test_can_load_data_as_pandas_dataframe(self):
    spreadsheet = self.client.create("test_sheet")
    spreadsheet.worksheet("Sheet1").update([
        ["header 1", "header 2"],
        ["row 1 col 1", "row 1 col 2"],
        ["row 2 col 1", "row 2 col 2"],
    ])

    google_sheet = sheets.GoogleSheet.load(spreadsheet.url)
    data = google_sheet["Sheet1"]

    expected_data = pd.DataFrame({
        "header 1": ["row 1 col 1", "row 2 col 1"],
        "header 2": ["row 1 col 2", "row 2 col 2"],
    })

    pd.testing.assert_frame_equal(data, expected_data)

  def test_can_load_data_with_index_as_pandas_dataframe(self):
    spreadsheet = self.client.create("test_sheet")
    spreadsheet.worksheet("Sheet1").update([
        ["my_index", "header 1", "header 2"],
        ["a", "row 1 col 1", "row 1 col 2"],
        ["b", "row 2 col 1", "row 2 col 2"],
    ])
    spreadsheet.worksheet("Sheet1").freeze(cols=1)  # Indexes are frozen columns

    google_sheet = sheets.GoogleSheet.load(spreadsheet.url)
    data = google_sheet["Sheet1"]

    expected_data = pd.DataFrame({
        "my_index": ["a", "b"],
        "header 1": ["row 1 col 1", "row 2 col 1"],
        "header 2": ["row 1 col 2", "row 2 col 2"],
    }).set_index("my_index")

    pd.testing.assert_frame_equal(data, expected_data)

  def test_writing_data_writes_to_sheet(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")
    data = pd.DataFrame({
        "my_index": ["a", "b"],
        "header 1": ["row 1 col 1", "row 2 col 1"],
        "header 2": ["row 1 col 2", "row 2 col 2"],
    }).set_index("my_index")

    google_sheet["Sheet1"] = data

    spreadsheet = self.client.open_by_url(google_sheet.url)
    worksheet = spreadsheet.worksheet("Sheet1")
    self.assertListEqual(
        worksheet._data,
        [
            ["my_index", "header 1", "header 2"],
            ["a", "row 1 col 1", "row 1 col 2"],
            ["b", "row 2 col 1", "row 2 col 2"],
        ],
    )

  def test_writing_data_updates_size_of_worksheet_to_match_data(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")
    spreadsheet = self.client.open_by_url(google_sheet.url)
    worksheet = spreadsheet.worksheet("Sheet1")

    self.assertEqual(worksheet.row_count, 1000)
    self.assertEqual(worksheet.col_count, 26)

    google_sheet["Sheet1"] = pd.DataFrame({
        "my_index": ["a", "b"],
        "header 1": ["row 1 col 1", "row 2 col 1"],
        "header 2": ["row 1 col 2", "row 2 col 2"],
    }).set_index("my_index")

    self.assertEqual(worksheet.row_count, 3)
    self.assertEqual(worksheet.col_count, 3)

  def test_writing_sets_format_of_column_names(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")

    google_sheet["Sheet1"] = pd.DataFrame({
        "my_index": ["a", "b"],
        "header 1": ["row 1 col 1", "row 2 col 1"],
        "header 2": ["row 1 col 2", "row 2 col 2"],
    }).set_index("my_index")

    spreadsheet = self.client.open_by_url(google_sheet.url)
    worksheet = spreadsheet.worksheet("Sheet1")

    for cell_format in worksheet._formatting[0]:
      # First row should be formatted like the header
      self.assertDictEqual(cell_format, sheets.HEADING_FORMAT)

    for row_format in worksheet._formatting[1:]:
      for cell_format in row_format:
        # All other rows should not be formatted
        self.assertDictEqual(cell_format, {})

  def test_writing_freezes_the_index_and_columns(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")
    spreadsheet = self.client.open_by_url(google_sheet.url)
    worksheet = spreadsheet.worksheet("Sheet1")

    self.assertEqual(worksheet.frozen_row_count, 0)
    self.assertEqual(worksheet.frozen_col_count, 0)

    google_sheet["Sheet1"] = pd.DataFrame({
        "my_index": ["a", "b"],
        "header 1": ["row 1 col 1", "row 2 col 1"],
        "header 2": ["row 1 col 2", "row 2 col 2"],
    }).set_index("my_index")

    self.assertEqual(worksheet.frozen_row_count, 1)
    self.assertEqual(worksheet.frozen_col_count, 1)

  def test_writing_data_adds_worksheet_if_needed(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")
    spreadsheet = self.client.open_by_url(google_sheet.url)

    worksheet_names = [sheet.title for sheet in spreadsheet.worksheets()]
    self.assertListEqual(worksheet_names, ["Sheet1"])

    google_sheet["Sheet2"] = pd.DataFrame({
        "my_index": ["a", "b"],
        "header 1": ["row 1 col 1", "row 2 col 1"],
        "header 2": ["row 1 col 2", "row 2 col 2"],
    }).set_index("my_index")

    worksheet_names = [sheet.title for sheet in spreadsheet.worksheets()]
    self.assertListEqual(worksheet_names, ["Sheet1", "Sheet2"])

  def test_writing_changed_data_overwrites_existing_data(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")
    google_sheet["Sheet1"] = pd.DataFrame({
        "my_index": ["a", "b", "c", "d"],
        "header 1": ["1", "2", "3", "4"],
        "header 2": ["5", "6", "7", "8"],
    }).set_index("my_index")

    # Some rows changed, some rows added, some rows unchanged
    new_data = pd.DataFrame({
        "my_index": ["a", "b change", "c", "d change", "e"],
        "header 1": ["1 change", "2", "3", "4 change", "9"],
        "header 2": ["5", "6 change", "7", "8", "10"],
    }).set_index("my_index")
    google_sheet["Sheet1"] = new_data

    pd.testing.assert_frame_equal(google_sheet["Sheet1"], new_data)

  def test_writing_data_with_different_columns_overwrites_existing_data(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")
    google_sheet["Sheet1"] = pd.DataFrame({
        "my_index": ["a", "b", "c", "d"],
        "header 1": ["1", "2", "3", "4"],
        "header 2": ["5", "6", "7", "8"],
    }).set_index("my_index")

    # Column name changed
    new_data = pd.DataFrame({
        "my_index": ["a", "b", "c", "d"],
        "header 1 changed": ["1", "2", "3", "4"],
        "header 2": ["5", "6", "7", "8"],
    }).set_index("my_index")
    google_sheet["Sheet1"] = new_data

    pd.testing.assert_frame_equal(google_sheet["Sheet1"], new_data)

  def test_delete_worksheet_deletes_the_worksheet(self):
    google_sheet = sheets.GoogleSheet.new("test_sheet")
    google_sheet["Sheet2"] = pd.DataFrame({
        "my_index": ["a", "b", "c", "d"],
        "header 1": ["1", "2", "3", "4"],
        "header 2": ["5", "6", "7", "8"],
    }).set_index("my_index")

    self.assertIn("Sheet2", google_sheet)
    google_sheet.delete_worksheet("Sheet2")
    self.assertNotIn("Sheet2", google_sheet)


if __name__ == "__main__":
  absltest.main()
