<img align="left"  height=230, src="./images/copycat_logo.png" alt="copycat_logo" />

# Copycat: AI-Generated Google Search Ad Copy That Matches Your Brand Style

[![python](https://img.shields.io/badge/Python->=3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/gtech-copycat?logo=pypi&logoColor=white&style=flat)](https://pypi.org/project/gtech-copycat/)
[![GitHub last commit](https://img.shields.io/github/last-commit/google-marketing-solutions/copycat)](https://github.com/google-marketing-solutions/copycat/commits)
[![Code Style: Google](https://img.shields.io/badge/code%20style-google-blueviolet.svg)](https://google.github.io/styleguide/pyguide.html)
[![Open in Colab](https://img.shields.io/badge/Graphical_User_Interface-blue?style=flat&logo=google%20colab&labelColor=grey)](https://colab.research.google.com/github/google-marketing-solutions/copycat/blob/main/copycat_ui_demo.ipynb)
[![Open in Colab](https://img.shields.io/badge/Python_Quickstart-blue?style=flat&logo=google%20colab&labelColor=grey)](https://colab.research.google.com/github/google-marketing-solutions/copycat/blob/main/copycat_demo.ipynb)

**Disclaimer: This is not an official Google product.**

<br />

[Overview](#overview) •
[Getting started](#getting-started) •
[How it works](#how-copycat-works) •
[Copycat UI walkthrough](#copycat-ui-walkthrough) •
[Example use-cases](#example-use-cases) •
[FAQs](#FAQs) •
[Citing Copycat](#citing-copycat)

## Overview

Copycat is a Python package that leverages Google Gemini models to generate high-quality ad copies for Google Search campaigns. Whether you need [Responsive Search Ads](https://support.google.com/google-ads/answer/12159142?hl=en-GB) or [Text Ads](https://support.google.com/google-ads/answer/1704389?hl=en-GB), Copycat learns from your top-performing ads and brand guidelines to create copy that seamlessly aligns with your brand voice.

**How it works:**

Copycat analyzes your existing high-quality search ads (headlines, descriptions, and keywords) to learn your brand style. You then provide new keywords, and Copycat generates corresponding headlines and descriptions that match your brand's tone and messaging.

**Benefits:**

* **Efficiency:**  Generate compelling ad copy quickly for various campaigns, saving time and resources.
* **Quality:** Maintain high ad standards that reflect your brand's unique style.
* **Scalability:** Expand your Google Ads reach without compromising quality or brand consistency.

<p align="center">
<img width="600" src="./images/copycat_simplified_architecture.png" alt="copycat_simplified_architecture" />
</p>

## Getting Started

Explore Copycat with these Colab notebooks:

* **`copycat_ui_demo.ipynb`:**  A user-friendly graphical interface, ideal for non-technical users and initial exploration.
    * Open in [Colab](https://colab.research.google.com/github/google-marketing-solutions/copycat/blob/main/copycat_ui_demo.ipynb)
    * View in [Github](https://github.com/google-marketing-solutions/copycat/blob/main/copycat_ui_demo.ipynb)

* **`copycat_demo.ipynb`:** A Python-based quick start example for users who want to integrate Copycat into other tools or build upon its functionality.
    * Open in [Colab](https://colab.research.google.com/github/google-marketing-solutions/copycat/blob/main/copycat_demo.ipynb)
    * View in [Github](https://github.com/google-marketing-solutions/copycat/blob/main/copycat_demo.ipynb)

**Prerequisites:**

* **Google Cloud Project:** You'll need a valid Google Cloud Project with the Vertex AI API enabled. Using these services will incur costs.
    * Model pricing: [See details](https://ai.google.dev/pricing)
    * Terms of service: [Review here](https://ai.google.dev/gemini-api/terms)

**Important:** Always review and edit generated ad copy before using it in your campaigns. Ensure compliance with [Google Ads policies](https://support.google.com/adspolicy/answer/6008942?hl=en).


## How Copycat Works

Copycat learns your brand style from your existing Google Search Ads to generate new ads for new keywords. Here's the process:

<img align="left", src="./images/copycat_prompt_diagram.png" alt="copycat_prompt_diagram" />

1. **Training Ads:** Provide examples of high-quality search ads (headlines, descriptions, and targeted keywords).  These "training ads" should come directly from your Google Ads account, we recommend selecting high quality ad copies (100 or more varied example ads is optimal, gut Copycat can work with fewer, see the FAQs). An [example Google Ads Script](https://github.com/google-marketing-solutions/copycat/blob/main/google_ads_scripts/ads_data_pull.js) is provided to help with this.

2. **Exemplar Ads:** Copycat uses [Affinity Propagation](https://scikit-learn.org/1.5/modules/clustering.html#affinity-propagation) to reduce the training ads to a smaller set of "exemplar ads." This ensures diversity and minimizes redundancy while capturing the different styles in your training data.

3. **Style Guide Generation:** Copycat uses Gemini to generate a style guide from the exemplar ads. You can also include PDF files with brand guidelines. Manually edit the style guide to refine it further.

4. **New Ad Generation:** Provide new keywords (or keywords from existing ad groups) for which you want to generate ads. You can also provide additional instructions to guide Copycat (e.g., "Mention we offer a 10% discount"). Copycat combines the style guide, your instructions, and relevant exemplar ads in a prompt to Gemini, which then generates the new ad copy.

5. **Ad Extension:** If you have existing headlines or descriptions, Copycat can fill in the remaining slots in your ads.

### Evaluation

Copycat evaluates generated ads:

* **Memorization Check:** Flags ads where all headlines and descriptions are taken directly from exemplar ads.
* **Style Similarity:** Measures the similarity between the generated ad and the closest exemplar ad.
* **Keyword Similarity:** Measures the similarity between the generated ad and the new keywords.

These metrics help you identify ads that need further refinement.

<p align="center">
<img src="./images/copycat_evaluation_diagram.png" alt="copycat_evaluation_diagram" />
</p>

## Copycat UI Walkthrough

The [Copycat UI](https://colab.research.google.com/github/google-marketing-solutions/copycat/blob/main/copycat_ui_demo.ipynb) provides an easy way to get started. Each project is backed by a Google Sheet for easy data management and continuation.

**Logging:** Copycat logs progress to the "Logs" tab in your Google Sheet. Adjust the logging level (INFO or ERROR) as needed. This is especially important for large-scale ad generation where you should set the logging level to ERROR to avoid too many logs being sent.

### Input Data

Your Google Sheet should have these tabs:

* **Training Ads:**  Existing ads with headlines, descriptions, and keywords (comma-separated list). One row per ad.
* **New Keywords:** New keywords with corresponding Campaign and Ad Group identifiers. One row per keyword.
* **Extra Instructions for New Ads (Optional):**  Provide extra context or instructions for specific ads, campaigns, or versions. One row per instruction.

**Demo Data:** Use the provided demo data for initial testing, but **do not use it for real campaigns.**

### Starting a New Project

1. Ensure you have a Google Cloud Project with the Vertex AI API enabled.
2. Open the Copycat UI in [Colab](https://colab.research.google.com/github/google-marketing-solutions/copycat/blob/main/copycat_ui_demo.ipynb).
3. Run the code cells sequentially.
4. Authenticate with Google Drive and GCP.
5. Create a new Google Sheet (with demo data for testing) or load an existing one.
6. Enter your GCP Project ID and location in the setup page.
7. **Prepare Data:**
    * Validate your sheet using the "Validate Sheet" button.
8. **New Copycat Instance:**
    * Configure parameters (company name, language, etc.). See FAQs for details.
    * Build the Copycat instance.
9. **Style Guide:**
    * Generate the style guide (optionally include PDF files with brand guidelines).
    * Review and edit the style guide.
10. **Generate Ads:**
    * Adjust parameters (memorization allowance, versions, etc.).
    * Preview the prompt.
    * Generate ads in batches.
11. **Review and Refine:**
    * Monitor progress in the Google Sheet ("Generated Ads" tab).
    * Edit generated ads as needed.
    * Use style and keyword similarity scores to identify ads for refinement.
12. **Upload to Google Ads:** Once satisfied, upload the generated ads to your Google Ads campaigns.

### Resuming an Existing Project

Use the URL of a previously created Copycat Google Sheet to continue your work.

## Example Use Cases

* **Generating ads for new keywords:** Quickly create ads for new or trending keywords.
* **Expanding existing ads:** Fill empty headline and description slots in existing Responsive Search Ads.
* **Rewriting existing ads:** Update your ads at scale to reflect changes in brand tone or messaging.

## FAQs

### 1. What do the parameters for creating a new Copycat instance in the UI mean?

* **Company Name**: The name of the company / brand you are advertising.
* **Ad Copy Language**: The language of the ads. This should be the language of the training ads. If you want to generate ads in a different language to the training ads, then you can add an additional instruction like "Generate this ad in language X".
* **Ad Format**: The ad format you want to generate for. This sets the number of headlines and descriptions to generate.
* **How to handle special variables**: Special variables like Dynamic Keyword Insertion (DKI) or Customizers can either be removed and replaced with their default values or left in the ads. If you remove them then Copycat won't ever see any examples of DKI or Customizers, so it won't generate them. If you keep them then make sure you explain in the style guide what DKI and Customizers are, otherwise Copycat won't understand them well and might generate nonsense DKI or Customizers.
* **How to handle invalid training ads**: Copycat will check that all your training ads have the expected number of headlines and descriptions, and that those headlines and descriptions have up to 30 and 90 characters long respectively. Any that don't meet the criteria can either be dropped (removed), skipped (left in the data) or you can raise an error.
* **Dimensionality**: This is the dimensionality of the embedding model. This can be up to 768, a smaller number will mean everything will run faster, but a higher dimensionality will understand the ads better and can lead to better results.
* **Batch Size**: This refers to the batch size of the embedding model. A higher batch size can run faster if you have lots of ads, but you can hit quota limits.
* **Exemplar selection method**: You can either select the Exemplar ads with Affinity Propagation or randomly. Affinity propagation is usually the best option. It creates clusters of your training ads and then selects a single exemplar from each cluster, ensuring that the exemplars are representative of all the different training ads but no two exemplars are too similar to each other. Selecting randomly can be useful if you have a very small number of ads to start with, because if you select random and the max exemplar ads is larger than the total number of ads you have, then it will use all your training ads as exemplar ads.
* **Max initial ads and max exemplar ads**: The max exemplar ads is the maximum number of exemplars you want to end up with. If using Affinity Propagation you also need to set the max initial ads. If you have more training ads that the max initial ads then first it will randomly select the max initial ads before running the affinity propagation, otherwise the affinity propagation will take a long time to run.
* **Custom affinity preference**: You can set this, which controls the number of exemplars selected when using affinity propagation. It must be a negative number, and the more negative the number the fewer exemplars will be selected.

### 2. What do the parameters for generating the style guide in the UI mean?

* **Chat Model**: The chat model to use to generate the style guide. Pro is higher quality but more expensive than flash.
* **Temperature, Top-K and Top-P**: These are standard AI model parameters that control the level of creativity the AI has when generating. Have a look [here](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values) for more information on these.
* **Use exemplar ads to generate style guide**: Whether to show the exemplar ads to Gemini when generating the style guide. If this is turned off you must provide supplementary materials via the google cloud bucket instead.
* **Additional instructions for generating the style guide**: This is a set of optional instructions to pass to Gemini when generating the style guide.
* **Google Cloud Bucket URI containing supplementary materials**: This is the URI containing the pdf files which will also be shown to gemini when generating the style guide. All of the files in this bucket will be used, so make sure it only contains the files you want to use for the style guide. This is optional, but if you are not using exemplar ads you must include it.

### 3. What do the parameters for generating new ads in the UI mean?

* **Chat Model**: The chat model to use to generate the style guide. Pro is higher quality but more expensive than flash.
* **Use style guide**: Whether or not to include the style guide in the prompt. If you don't include it then Gemini will infer the style only from the examples it is shown.
* **N in-context examples**: How many exemplar ads do you want to include in the prompt.
* **Temperature, Top-K and Top-P**: These are standard AI model parameters that control the level of creativity the AI has when generating. Have a look [here](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values) for more information on these.
* **Allow memorised headlines / descriptions**: Copycat always checks whether the headlines or descriptions were memorised from the training data. If these are set to off, then memorised headlines and descriptions are not allowed and if there are any they will cause that add to be classed as unsuccessful. If you choose to allow memorised headlines or descriptions then Copycat will still flag memorizations but they will not cause the generation to be flagged as unsuccessful.
* **Number of versions per ad group**: How many versions of each ad to generate. This lets you generate multiple versions of the ads for each set of keywords, perhaps with different additional instructions for each version.

### 4. Copycat is ignoring some parts of my Style Guide, what should I do?

Try changing the way the style guide is written to make sure it's clear. Remove any redundant information from the style guide. You can also try reminding Copycat by adding an additional instruction with the bits of the style it is missing.

If this is a part of the style guide that is new and so your example ads don't follow it, then try reducing the number of in context examples you use when generating the ads.

### 5. How many training ads to I need for Copycat to learn from?

Copycat can work with anything from 10s to 1000s of ads. More is better but it's best to just experiment with what you have. Only include high quality ads, there is no benefit in including poor quality ads to increase the number of training ads you have. The fewer training ads you have, the more Copycat will need to rely on the stlye guide, so make sure that is good and captures all the relevant information.

### 6. I don't have any high quality existing ads for Copycat to learn from, can I still use it?

You can, but you need to make sure you have some supplementary materials to learn the style guide from. You can then tell Copycat not to use the Exemplar ads in the style guide, and set the number of in context examples to 0 when generating the ads.

You will need to include one dummy row of training ads otherwise Copycat won't run, but as long as you follow these instructions it won't be used.

### 7. It's difficult to explain my brand style in words in the style guide, what should I do?

In this case Copycat needs to rely primarily on the in context examples to learn your style implicitly. Make sure you have lots of exemplar ads and increase the number of in context examples included at generation time.

### 8. Can you provide more information on the input data for the UI?

The input google sheet for the UI requires 3 tabs of data. It's best to create a new sheet with the UI and then copy and paste the data in, this will ensure that the columns are all named correctly. Below is an explanation of what needs to be in each tab.

If you are just getting started and want to demo how Copycat works, you can just use the demo data that is automatically added to the google sheet by Copycat when you create it, so if you plan to do that you can skip this section for now. **However you must not use the demo data when generating real ads you plan to use in Google Ads campaigns, you must use your own data.**

#### Training Ads

This contains the existing ads that Copycat will learn your style from. Copycat will learn how to write good ad copy based on these ads, so make sure you only include good quality ads here. You can include as many ads as you have, Copycat will automatically select relevant ones from the ones you provide here.

The data should have one row per existing ad. Each ad must contain the keywords it was used for, as a comma separated list, and then the headlines and descriptions in different columns named "Headline 1", "Headline 2" ..., "Description 1", "Description 2"... By default Copycat will create a sheet for RSA ads, so it will have 15 headline slots and 4 description slots, but if you have fewer than that you can delete the redundant columns.

#### New Keywords

This tab contains the new keywords you want to generate new ads for. In this tab you should have a single row per keyword, and include the Campaign and Ad Group as identifiers. Copycat will automatically group all the keywords together for the same campaign and ad group when generating new ads.

If you don't know the name of the campaign or ad group yet just use a random identifier. The name is not used by Copycat when generating the ads, it is just used to collect all of the keywords together to generate a single ad per ad group.

#### Extra Instructions for New Ads

This is optional, and you can leave it blank. But if you want to you can add any extra instructions you might want to provide for the ads. You can use the `__ALL__` placeholder for either Campaign ID, Ad Group or Version and then that instruction will apply to all of the values for that column. This allows you to create combinations of instructions, where some are generic (for all ads), and some are specific (a single ad, a campaign, or a version). Note: when generating ads, Copycat will let you choose how many versions you want to generate for each ad group. This is what “version” is referring to. So if you select 3 versions when generating, you can have a different instruction for each version if you want to.

These instructions are inserted at the end of the prompt, after the style guide and all of the examples, so it can be used to instruct Copycat to deviate from the Style Guide and previous ads if required. For example, even if all your training ads are in English, you could add an additional instruction like "Please write the next ad in German" to generate a german version of an ad.

## Citing Copycat

To cite this repository:

```
@software{copycat_github,
  author = {Sam Bailey, Piet Snel, Christiane Ahlheim, Sumedha Menon, Hector Parra, Jaime Martínez, Letizia Bertolaja},
  title = {Copycat: AI generated Google Search ad copy that matches your brand style},
  url = {https://github.com/google-marketing-solutions/copycat},
  version = {0.0.2},
  year = {2024},
}
```
