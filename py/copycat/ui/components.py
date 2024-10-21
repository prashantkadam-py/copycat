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

"""Reusable UI components for Copycat."""

import mesop as me

from copycat.ui import styles


@me.content_component
def row(gap: int = styles.DEFAULT_ROW_AND_COLUMN_GAP, **kwargs):
  """Creates a row of UI elements."""
  with me.box(
      style=me.Style(display="flex", flex_direction="row", gap=gap, **kwargs)
  ):
    me.slot()


@me.content_component
def column(gap: int = styles.DEFAULT_ROW_AND_COLUMN_GAP, **kwargs):
  """Creates a column of UI elements."""
  with me.box(
      style=me.Style(display="flex", flex_direction="column", gap=gap, **kwargs)
  ):
    me.slot()


@me.content_component
def header_bar(**kwargs):
  """Creates a header bar."""
  with me.box(
      style=me.Style(
          background=me.theme_var("surface-container"),
          padding=me.Padding.all(10),
          align_items="center",
          display="flex",
          gap=5,
          justify_content="space-between",
          **kwargs
      )
  ):
    me.slot()


@me.content_component
def header_section():
  """Adds a section to the header."""
  with me.box(style=me.Style(**styles.HEADER_SECTION_STYLE)):
    me.slot()


@me.content_component
def conditional_tooltip(
    disabled: bool,
    disabled_tooltip: str = "",
    enabled_tooltip: str = "",
    **kwargs
):
  """Adds a tooltip to a UI element depending on whether it is disabled."""
  if disabled and disabled_tooltip:
    with me.tooltip(message=disabled_tooltip, **kwargs):
      me.slot()
  elif not disabled and enabled_tooltip:
    with me.tooltip(message=enabled_tooltip, **kwargs):
      me.slot()
  else:
    me.slot()


@me.content_component
def rounded_box_section(title: str = "", **kwargs):
  """Adds a rounded box section with an optional title."""
  with me.box(style=me.Style(**(styles.ROUNDED_BOX_SECTION_STYLE | kwargs))):
    if title:
      me.text(title, type=styles.ROUNDED_BOX_SECTION_HEADER_TYPE)
    me.slot()
