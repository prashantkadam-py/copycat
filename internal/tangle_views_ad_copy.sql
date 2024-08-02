-- Copyright 2024 Google LLC.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     https://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

/**
 * Pull Ad copy and keywords for a given customer ID and campaign IDs from Tangle Views.
 *
 * Parameters:
 *   CAMPAIGN_IDS: The campaign IDs to pull the data for. For example:
 *     11111111111,22222222222
 *   CUSTOMER_IDS: The customer IDs to pull the data for. For example:
 *     'customer_id: 1112223333', 'customer_id: 4445556666'
 */

-- How many days to look back when calculating the number of impressions. The report only pulls data
-- that has had at least one impression in the last N days.
DEFINE MACRO LOOKBACK_DAYS 30;


-- The Ad Copy & corresponding ad group and campaign information.
DEFINE TABLE RsaAssetDetails (
  format = 'tangle',
  query = '''
    SELECT
      Customer.customer_id AS customer_id,
      Customer.descriptive_name AS account_name,
      AdGroupAd.campaign_id AS campaign_id,
      Campaign.name AS campaign_name,
      AdGroupAd.ad_group_id AS ad_group_id,
      AdGroup.name AS ad_group_name,
      AdGroupAd.ad.ad_id AS ad_id,
      AdGroupAd.ad.type AS ad_type,
      AdGroupAd.ad_strength AS ad_strength,
      AdGroupAd.ad.final_urls AS final_urls,
      AssetGroup.asset_group_id AS asset_group_id,
      AssetGroup.served_assets AS served_assets,
      AssetGroup.asset_data AS asset_data
    WHERE
      AdGroupAd.campaign_id in (${CAMPAIGN_IDS})
      AND DayV2.day >= ${YYYYMMDD-$LOOKBACK_DAYS}
      AND DayV2.day <= ${YYYYMMDD-1}
    HAVING
      impressions > 0
  ''',
  rootids_protos = [${CUSTOMER_IDS}]);


-- The keyword data
DEFINE TABLE Keywords (
  format = 'tangle',
  query = '''
    SELECT
      Customer.customer_id AS customer_id,
      Campaign.campaign_id AS campaign_id,
      AdGroupAd.ad_group_id AS ad_group_id,
      AdGroupCriterion.status as keyword_status,
      AdGroupCriterion.keyword.text AS keyword_text
    WHERE
      Campaign.campaign_id in (${CAMPAIGN_IDS})
      AND DayV2.day >= ${YYYYMMDD-$LOOKBACK_DAYS}
      AND DayV2.day <= ${YYYYMMDD-1}
    HAVING
      impressions > 0
  ''',
  rootids_protos = [${CUSTOMER_IDS}]);

WITH
  AssetText AS (
    SELECT DISTINCT
      asset_group_id,
      asset_info.asset.asset_id AS asset_id,
      asset_info.asset.text_asset.text AS text
    FROM
      RsaAssetDetails,
      RsaAssetDetails.asset_data.asset_with_read_only_data AS asset_info
  ),
  AssetUsage AS (
    SELECT DISTINCT
      asset_group_id,
      asset_usages.asset_id AS asset_id,
      CAST(asset_usages.served_asset_field_type AS STRING) AS field_type
    FROM
      RsaAssetDetails,
      RsaAssetDetails.served_assets.usages AS asset_usages
  ),
  AssetMapping AS (
    SELECT DISTINCT
      ast.asset_group_id,
      ast.asset_id,
      asu.field_type AS field_type,
      ast.text
    FROM AssetText AS ast
    INNER JOIN AssetUsage AS asu
      USING (asset_group_id, asset_id)
    ORDER BY asset_group_id ASC, field_type ASC
  ),
  AssetTexts AS (
    SELECT
      asset_group_id,
      asset_id,
      field_type AS ad_copy_type,
      text AS ad_copy_text,
    FROM AssetMapping
    WHERE field_type LIKE 'HEADLINE_%' OR field_type LIKE 'DESCRIPTION_%'
  ),
  CombinedResults AS (
    SELECT
      customer_id,
      account_name,
      campaign_id,
      campaign_name,
      ad_group_id,
      ad_group_name,
      ad_id,
      CAST(ad_type AS global_proto_db.`ads_enums.CreativeTypePB.Enum`) AS ad_type,
      ad_strength,
      ad_copy_type,
      ad_copy_text,
      STRING_AGG(DISTINCT final_url ORDER BY final_url) AS final_urls,
      STRING_AGG(DISTINCT keyword_text ORDER BY keyword_text) AS keywords,
    FROM
      RsaAssetDetails,
      UNNEST(final_urls) AS final_url
    INNER JOIN AssetTexts
      USING (asset_group_id)
    INNER JOIN Keywords
      USING (customer_id, campaign_id, ad_group_id)
    WHERE
      keyword_status = 'Status_Active'
    GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ORDER BY
      customer_id,
      campaign_id,
      ad_group_id,
      ad_id,
      ad_copy_type
  )
SELECT *
FROM
  CombinedResults
    PIVOT(
      ANY_VALUE(ad_copy_text)
        FOR
          ad_copy_type IN (
            'HEADLINE_1',
            'HEADLINE_2',
            'HEADLINE_3',
            'HEADLINE_4',
            'HEADLINE_5',
            'HEADLINE_6',
            'HEADLINE_7',
            'HEADLINE_8',
            'HEADLINE_9',
            'HEADLINE_10',
            'HEADLINE_11',
            'HEADLINE_12',
            'HEADLINE_13',
            'HEADLINE_14',
            'HEADLINE_15',
            'DESCRIPTION_1',
            'DESCRIPTION_2',
            'DESCRIPTION_3',
            'DESCRIPTION_4'));
