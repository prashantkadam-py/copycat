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

import os

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from copycat.data import utils


class ExplodeAndCollapseHeadlinesAndDescriptionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="with_headlines_and_descriptions",
          data=pd.DataFrame({
              "Headline 1": ["a", "b"],
              "Headline 2": ["c", "--"],
              "Headline 3": ["d", ""],
              "Description 1": ["e", "f"],
              "Description 2": ["g", "--"],
              "Other column": [1, 2],
          }),
          expected=pd.DataFrame({
              "headlines": [["a", "c", "d"], ["b"]],
              "descriptions": [["e", "g"], ["f"]],
              "Other column": [1, 2],
          }),
      ),
      dict(
          testcase_name="with_no_headlines",
          data=pd.DataFrame({
              "Description 1": ["e", "f"],
              "Description 2": ["g", "--"],
              "Other column": [1, 2],
          }),
          expected=pd.DataFrame({
              "headlines": [[], []],
              "descriptions": [["e", "g"], ["f"]],
              "Other column": [1, 2],
          }),
      ),
      dict(
          testcase_name="with_no_descriptions",
          data=pd.DataFrame({
              "Headline 1": ["a", "b"],
              "Headline 2": ["c", "--"],
              "Headline 3": ["d", ""],
              "Other column": [1, 2],
          }),
          expected=pd.DataFrame({
              "headlines": [["a", "c", "d"], ["b"]],
              "descriptions": [[], []],
              "Other column": [1, 2],
          }),
      ),
      dict(
          testcase_name="with_no_headlines_or_descriptions",
          data=pd.DataFrame({"Other column": [1, 2]}),
          expected=pd.DataFrame({
              "headlines": [[], []],
              "descriptions": [[], []],
              "Other column": [1, 2],
          }),
      ),
  )
  def test_collapse_headlines_and_descriptions(self, data, expected):
    actual = utils.collapse_headlines_and_descriptions(data)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)

  @parameterized.named_parameters(
      dict(
          testcase_name="with_headlines_and_descriptions",
          data=pd.DataFrame({
              "headlines": [["a", "c", "d"], ["b"]],
              "descriptions": [["e", "g"], ["f"]],
              "Other column": [1, 2],
          }),
          expected=pd.DataFrame({
              "Headline 1": ["a", "b"],
              "Headline 2": ["c", "--"],
              "Headline 3": ["d", "--"],
              "Description 1": ["e", "f"],
              "Description 2": ["g", "--"],
              "Other column": [1, 2],
          }),
      ),
      dict(
          testcase_name="with_no_headlines",
          data=pd.DataFrame({
              "headlines": [[], []],
              "descriptions": [["e", "g"], ["f"]],
              "Other column": [1, 2],
          }),
          expected=pd.DataFrame({
              "Description 1": ["e", "f"],
              "Description 2": ["g", "--"],
              "Other column": [1, 2],
          }),
      ),
      dict(
          testcase_name="with_no_descriptions",
          data=pd.DataFrame({
              "headlines": [["a", "c", "d"], ["b"]],
              "descriptions": [[], []],
              "Other column": [1, 2],
          }),
          expected=pd.DataFrame({
              "Headline 1": ["a", "b"],
              "Headline 2": ["c", "--"],
              "Headline 3": ["d", "--"],
              "Other column": [1, 2],
          }),
      ),
      dict(
          testcase_name="with_no_headlines_or_descriptions",
          data=pd.DataFrame({
              "headlines": [[], []],
              "descriptions": [[], []],
              "Other column": [1, 2],
          }),
          expected=pd.DataFrame({"Other column": [1, 2]}),
      ),
  )
  def test_explode_headlines_and_descriptions(self, data, expected):
    actual = utils.explode_headlines_and_descriptions(data)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)

  def test_explode_headlines_and_descriptions_raises_value_error_if_index_not_unique(
      self,
  ):
    data = pd.DataFrame({
        "headlines": [["a", "c", "d"], ["b"]],
        "descriptions": [["e", "g"], ["f"]],
        "Other column": [1, 1],
    }).set_index("Other column")
    with self.assertRaises(ValueError):
      utils.explode_headlines_and_descriptions(data)

  @parameterized.parameters(
      "headlines",
      "descriptions",
  )
  def test_explode_headlines_and_descriptions_raises_value_error_if_headlines_or_descriptions_are_not_lists(
      self, column_name
  ):
    data = pd.DataFrame({
        "headlines": [["a", "c", "d"], ["b"]],
        "descriptions": [["e", "g"], ["f"]],
        "Other column": [1, 1],
    })
    data[column_name] = ["a", "b"]  # Column does not contain lists.
    with self.assertRaises(ValueError):
      utils.explode_headlines_and_descriptions(data)


class IterateOverBatchesTest(parameterized.TestCase):

  def test_iterate_over_batches(self):
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
    batches = list(utils.iterate_over_batches(data, batch_size=3))

    self.assertLen(batches, 3)
    pd.testing.assert_frame_equal(batches[0], data.iloc[:3])
    pd.testing.assert_frame_equal(batches[1], data.iloc[3:6])
    pd.testing.assert_frame_equal(batches[2], data.iloc[6:7])

  def test_iterate_over_batches_with_limit_rows(self):
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
    batches = list(utils.iterate_over_batches(data, batch_size=3, limit_rows=5))

    self.assertLen(batches, 2)
    pd.testing.assert_frame_equal(batches[0], data.iloc[:3])
    pd.testing.assert_frame_equal(batches[1], data.iloc[3:5])

  def test_iterate_over_batches_with_too_large_limit_rows(self):
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
    batches = list(
        utils.iterate_over_batches(data, batch_size=3, limit_rows=20)
    )

    self.assertLen(batches, 3)
    pd.testing.assert_frame_equal(batches[0], data.iloc[:3])
    pd.testing.assert_frame_equal(batches[1], data.iloc[3:6])
    pd.testing.assert_frame_equal(batches[2], data.iloc[6:7])


if __name__ == "__main__":
  absltest.main()
