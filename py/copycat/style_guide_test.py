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

from unittest.mock import MagicMock, patch

from absl.testing import absltest
from google.cloud import storage
from vertexai import generative_models

from copycat import style_guide
from copycat import testing_utils


class TestStyleGuideGenerator(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.test_response = generative_models.GenerationResponse.from_dict({
        "candidates": [
            {"content": {"parts": [{"text": "This is a test style guide"}]}}
        ]
    })

  def test_get_all_files(self):
    mock_storage_client = MagicMock(spec=storage.Client)
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.name = "test_file.pdf"
    mock_blob.content_type = "application/pdf"

    mock_storage_client.bucket.return_value = mock_bucket
    mock_bucket.list_blobs.return_value = [mock_blob]

    with patch("google.cloud.storage.Client", return_value=mock_storage_client):
      generator = style_guide.StyleGuideGenerator()
      result = generator.get_all_files(bucket_name="test_bucket")

    expected_result = [{
        "uri": "gs://test_bucket/test_file.pdf",
        "mime_type": "application/pdf",
    }]
    self.assertEqual(result, expected_result)

    mock_storage_client.bucket.assert_called_once_with("test_bucket")
    mock_bucket.list_blobs.assert_called_once()

  def test_generate_style_guide(self):
    """Tests the style guide generation process with a mock model."""

    with testing_utils.PatchGenerativeModel(
        response=self.test_response
    ) as model_patcher:

      generator = style_guide.StyleGuideGenerator()
      generator.file_info = [
          {"uri": "gs://test_bucket/file1.pdf", "mime_type": "application/pdf"}
      ]

      response = generator.generate_style_guide(
          brand_name="Test Brand",
          additional_style_instructions="Write in a fun and friendly tone.",
          model_name="gemini-1.5-pro-preview-0514",
          temperature=0.8,
      )

      # Assertions
      self.assertEqual(len(response.candidates), 1)  # Check for one candidate
      self.assertEqual(
          response.candidates[0].content.text,
          self.test_response.candidates[0].content.text,
      )  # Compare with the test response

      model_patcher.mock_generative_model.generate_content.assert_called_once()

if __name__ == "__main__":
  absltest.main()
