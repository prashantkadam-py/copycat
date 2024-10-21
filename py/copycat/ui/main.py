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

"""The main entrypoint for the Copycat UI."""

import mesop as me

from copycat.ui import components
from copycat.ui import setup_page
from copycat.ui import states
from copycat.ui import styles
from copycat.ui import sub_pages


all_sub_pages = sub_pages.SubPages()
all_sub_pages.add_page(
    setup_page.setup,
    nav_text="Setup",
    nav_icon="settings",
)


@me.page(path="/")
def home():
  """The home page of the Copycat UI.

  This contains all the scaffolding for the UI, but the actual content is
  rendered by the sub-pages.
  """
  state = me.state(states.AppState)

  with me.box(
      style=me.Style(
          display="grid", grid_template_rows="auto 1fr auto", height="100%"
      )
  ):

    # The header bar, containing the Copycat name and the Google Sheet URL.
    with components.header_bar(
        border=me.Border.symmetric(vertical=styles.DEFAULT_BORDER_STYLE)
    ):
      with components.header_section():
        me.text(
            "Copycat",
            type="headline-3",
            style=me.Style(margin=me.Margin(bottom=0)),
        )

      with components.header_section():
        me.text(
            "Google Sheet URL",
            type="headline-6",
            style=me.Style(margin=me.Margin(bottom=0)),
        )
        me.input(
            label="URL",
            value=state.google_sheet_url,
            type="url",
            appearance="outline",
            style=me.Style(
                width=500,
                padding=me.Padding.all(0),
                margin=me.Margin(top=20),
            ),
            readonly=True,
        )

      with components.header_section():
        pass

    # Google Sheet and Body
    with components.row(
        gap=0,
        height="100%",
        width="100%",
    ):
      all_sub_pages.render(
          height="100%",
          width="50%" if state.display_google_sheet else "100%",
          gap=0,
          border=me.Border(right=styles.DEFAULT_BORDER_STYLE),
      )

    # Google Sheet
    if state.display_google_sheet:
      with me.box(
          style=me.Style(
              height="100%",
              width="50%",
          )
      ):
        me.embed(
            src=state.google_sheet_url,
            style=me.Style(width="100%", height="100%"),
        )
