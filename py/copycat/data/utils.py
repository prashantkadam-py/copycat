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

"""Utility functions for working with data."""

from typing import Any, Callable, Generator

import pandas as pd


def collapse_headlines_and_descriptions(
    data: pd.DataFrame,
) -> pd.DataFrame:
  """Collapses headline and description columns into two new columns.

  Assumes that the headline and description column names have the format
  "Headline {i}" and "Description {i}", where {i} is the headline number, with
  a single headline or description per column.

  Then collapses them into two new columns, "headlines" and "descriptions",
  containing lists of headlines and descriptions respectively.

  Args:
    data: The input DataFrame.

  Returns:
    A new DataFrame with the headline and description columns collapsed into two
    new columns, "headlines" and "descriptions".
  """
  output_data = data.copy()

  headline_cols = [
      c
      for c in output_data.columns
      if c.startswith("Headline ")
      and c.split(" ")[1].isdigit()
      and len(c.split(" ")) == 2
  ]
  description_cols = [
      c
      for c in output_data.columns
      if c.startswith("Description ")
      and c.split(" ")[1].isdigit()
      and len(c.split(" ")) == 2
  ]
  output_data["headlines"] = pd.Series(
      {
          k: list(
              filter(lambda x: x != "--" and x, v),
          )
          for k, v in output_data[headline_cols].T.to_dict("list").items()
      },
      index=output_data.index,
  )
  output_data["descriptions"] = pd.Series(
      {
          k: list(filter(lambda x: x != "--" and x, v))
          for k, v in output_data[description_cols].T.to_dict("list").items()
      },
      index=output_data.index,
  )

  output_data = output_data.drop(columns=headline_cols + description_cols)

  return output_data


def _explode_to_columns(output_name: str) -> Callable[[list[Any]], pd.Series]:
  """Returns a function that explodes a list into a Series of columns.

  The returned function takes a list as input and returns a Series where each
  element of the list is assigned to a separate column, with the column names
  having the format f"{output_name} {i}", where i is the index of the element
  in the list and starts at 1.

  Args:
    output_name: The name of the output columns.

  Returns:
    A function that explodes a list into a Series of columns.
  """

  def apply_explode_to_columns(list_col: list[Any]) -> pd.Series:
    if not isinstance(list_col, list):
      raise ValueError(
          "The input to the explode_to_columns function must be a list, got"
          f" {type(list_col)} instead."
      )
    return pd.Series(
        list_col,
        index=[f"{output_name} {i+1}" for i in range(len(list_col))],
    )

  return apply_explode_to_columns


def explode_headlines_and_descriptions(data: pd.DataFrame) -> pd.DataFrame:
  """Explodes headline and description columns into separate columns.

  Assumes that the headline and description columns have been collapsed into two
  new columns, "headlines" and "descriptions", containing lists of headlines and
  descriptions respectively.

  Then explodes them into separate columns, with the column names having the
  format "Headline {i}" and "Description {i}", where {i} is the index of the
  headline or description and starts at 1.

  Args:
    data: The input DataFrame.

  Returns:
    A new DataFrame with the headline and description columns exploded into
    separate columns.

  Raises:
    ValueError: If the index of the input data is not unique.
  """
  if not data.index.is_unique:
    raise ValueError(
        "The index of the input data is not unique, cannot explode headlines"
        " and descriptions."
    )

  if "headlines" in data:
    headlines = (
        data["headlines"].apply(_explode_to_columns("Headline")).fillna("--")
    )
  else:
    headlines = pd.DataFrame()

  if "descriptions" in data:
    descriptions = (
        data["descriptions"]
        .apply(_explode_to_columns("Description"))
        .fillna("--")
    )
  else:
    descriptions = pd.DataFrame()

  output_data = (
      data.copy()
      .drop(columns=["headlines", "descriptions"])
      .merge(headlines, left_index=True, right_index=True)
      .merge(descriptions, left_index=True, right_index=True)
  )

  return output_data


def iterate_over_batches(
    data: pd.DataFrame, batch_size: int, limit_rows: int | None = None
) -> Generator[pd.DataFrame, None, None]:
  """Iterates over batches of data.

  Args:
    data: The input DataFrame.
    batch_size: The size of each batch.
    limit_rows: The maximum number of rows to iterate over. If None, all rows
      will be iterated over.

  Yields:
    A generator that yields batches of data.
  """
  if limit_rows is None or limit_rows > len(data):
    limit_rows = len(data)

  n_regular_batches = limit_rows // batch_size
  final_batch_size = limit_rows % batch_size

  for i in range(n_regular_batches):
    yield data.iloc[i * batch_size : (i + 1) * batch_size]

  if final_batch_size:
    yield data.iloc[
        n_regular_batches * batch_size : n_regular_batches * batch_size
        + final_batch_size
    ]
