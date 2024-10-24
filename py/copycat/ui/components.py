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
def row(gap: int = styles.DEFAULT_ROW_AND_COLUMN_GAP, **kwargs) -> None:
  """Creates a row of UI elements."""
  with me.box(
      style=me.Style(display="flex", flex_direction="row", gap=gap, **kwargs)
  ):
    me.slot()


@me.content_component
def column(gap: int = styles.DEFAULT_ROW_AND_COLUMN_GAP, **kwargs) -> None:
  """Creates a column of UI elements."""
  with me.box(
      style=me.Style(display="flex", flex_direction="column", gap=gap, **kwargs)
  ):
    me.slot()


@me.content_component
def header_bar(**kwargs) -> None:
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
def header_section() -> None:
  """Adds a section to the header."""
  with me.box(style=me.Style(**styles.HEADER_SECTION_STYLE)):
    me.slot()


@me.content_component
def conditional_tooltip(
    disabled: bool,
    disabled_tooltip: str = "",
    enabled_tooltip: str = "",
    **kwargs
) -> None:
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
def rounded_box_section(title: str = "", **kwargs) -> None:
  """Adds a rounded box section with an optional title."""
  with me.box(style=me.Style(**(styles.ROUNDED_BOX_SECTION_STYLE | kwargs))):
    if title:
      me.text(title, type=styles.ROUNDED_BOX_SECTION_HEADER_TYPE)
    me.slot()


@me.content_component
def dialog(is_open: bool) -> None:
  """Renders a dialog component.

  The dialog is a full screen overlay that can be used to show additional
  information to the user.

  Args:
    is_open: Whether the dialog is visible or not.
  """
  with me.box(
      style=me.Style(
          background="rgba(0, 0, 0, 0.4)"
          if me.theme_brightness() == "light"
          else "rgba(255, 255, 255, 0.4)",
          display="block" if is_open else "none",
          height="100%",
          overflow_x="auto",
          overflow_y="auto",
          position="fixed",
          width="100%",
          z_index=1000,
      )
  ):
    with me.box(
        style=me.Style(
            align_items="center",
            display="grid",
            height="100vh",
            justify_items="center",
        )
    ):
      with me.box(
          style=me.Style(
              background=me.theme_var("surface-container-lowest"),
              border_radius=20,
              box_sizing="content-box",
              box_shadow=(
                  "0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px"
                  " #0000001f"
              ),
              margin=me.Margin.symmetric(vertical="0", horizontal="auto"),
              padding=me.Padding.all(20),
          )
      ):
        me.slot()
