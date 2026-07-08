"""
rename_columns.py
=================
Drop-in renaming block for merge_all.py.
Insert after step 8 (merge) and before step 9 (diagnostics).

Renames raw ACS Census variable codes and CDC PLACES column names to
human-readable labels. ACS columns that duplicate base-table derived
columns are suffixed with _acs to keep both.
"""

# ── ACS rename map ────────────────────────────────────────────────────────────
# Columns that overlap with base-table derived columns get an _acs suffix.
# All others get a clean human-readable name.

ACS_RENAME = {
    # ── Income / poverty ──────────────────────────────────────────────────────
    "B19013_001E": "Median Household Income_acs",      # duplicates median_income
    "B17001_002E": "Population Below Poverty_acs",     # duplicates pct_poverty (raw count)
    "B17001_001E": "Population Poverty Universe",
    "B22003_002E": "Households on SNAP_acs",           # duplicates pct_snap (raw count)
    "B22003_001E": "Households SNAP Universe",
    "B19057_002E": "Households with Public Assistance Income",
    "B19057_001E": "Households Public Assistance Universe",

    # ── Population / age ──────────────────────────────────────────────────────
    "B01003_001E": "Total Population_acs",             # duplicates population
    "B01002_001E": "Median Age_acs",                   # median age

    # ── Transportation ────────────────────────────────────────────────────────
    "B08201_002E": "Households No Vehicle_acs",        # duplicates pct_no_vehicle (raw count)
    "B08201_001E": "Households Vehicle Universe",
    "B08301_001E": "Total Workers Commuting",
    "B08301_010E": "Workers Using Public Transit_acs", # duplicates pct_transit (raw count)

    # ── Housing ───────────────────────────────────────────────────────────────
    "B25003_001E": "Total Occupied Housing Units",
    "B25003_002E": "Owner-Occupied Housing Units",
    "B25058_001E": "Median Contract Rent",
    "B25077_001E": "Median Home Value",
    "B25070_010E": "Households Rent 40-49% of Income",
    "B25070_011E": "Households Rent 50%+ of Income",
    "B25070_001E": "Households Rent Burden Universe",
    "B25002_003E": "Vacant Housing Units",
    "B25002_001E": "Total Housing Units",

    # ── Education ─────────────────────────────────────────────────────────────
    "B15003_017E": "Population High School Graduate",
    "B15003_022E": "Population Bachelor's Degree",
    "B15003_023E": "Population Master's Degree",
    "B15003_024E": "Population Professional Degree",
    "B15003_025E": "Population Doctorate Degree",
    "B15003_001E": "Population Education Universe",

    # ── Race / ethnicity ──────────────────────────────────────────────────────
    "B02001_002E": "Population White Alone",
    "B02001_003E": "Population Black or African American",
    "B02001_005E": "Population Asian Alone",
    "B03001_003E": "Population Hispanic or Latino",
    "B03002_003E": "Population Non-Hispanic White",
    "B03002_004E": "Population Non-Hispanic Black",
    "B03002_006E": "Population Non-Hispanic Asian",
    "B03002_012E": "Population Hispanic (Race Universe)",
    "B03002_001E": "Population Race/Ethnicity Universe",

    # ── Household structure ───────────────────────────────────────────────────
    "B11012_010E": "Single-Parent Female Households",
    "B11012_001E": "Total Households",

    # ── Internet access ───────────────────────────────────────────────────────
    "B28002_013E": "Households No Internet Access",
    "B28002_001E": "Households Internet Universe",

    # ── Employment ────────────────────────────────────────────────────────────
    "B23025_005E": "Civilian Unemployed",
    "B23025_001E": "Civilian Labor Force Universe",

}

# ── Age / Sex breakdown (B01001) ───────────────────────────────────────────
ACS_RENAME.update({
    # Male age bands
    "B01001_020E": "Male Population 65-66",
    "B01001_021E": "Male Population 67-69",
    "B01001_022E": "Male Population 70-74",
    "B01001_023E": "Male Population 75-79",
    "B01001_024E": "Male Population 80-84",
    "B01001_025E": "Male Population 85+",

    # Female 65+ bands
    "B01001_044E": "Female Population 65-66",
    "B01001_045E": "Female Population 67-69",
    "B01001_046E": "Female Population 70-74",
    "B01001_047E": "Female Population 75-79",
    "B01001_048E": "Female Population 80-84",
    "B01001_049E": "Female Population 85+",

    # Universe totals
    "B01001_001E": "Total Population Sex by Age Universe",
    "B01001_002E": "Total Male Population",
    "B01001_026E": "Total Female Population",
})

# ── PLACES rename map ─────────────────────────────────────────────────────────
# Point estimates only — 95CI columns are renamed for clarity but kept.
# Remove the CI entries below to drop those columns entirely.

PLACES_RENAME = {
    "TotalPopulation":          "PLACES Total Population",
    "TotalPop18plus":           "PLACES Population 18+",

    # Health behaviors
    "BINGE_CrudePrev":          "Binge Drinking % (Adults)",
    "BINGE_Crude95CI":          "Binge Drinking 95% CI",
    "CSMOKING_CrudePrev":       "Current Smoking % (Adults)",
    "CSMOKING_Crude95CI":       "Current Smoking 95% CI",
    "LPA_CrudePrev":            "Physical Inactivity % (Adults)",
    "LPA_Crude95CI":            "Physical Inactivity 95% CI",
    "SLEEP_CrudePrev":          "Short Sleep Duration % (Adults)",
    "SLEEP_Crude95CI":          "Short Sleep Duration 95% CI",

    # Chronic conditions
    "ARTHRITIS_CrudePrev":      "Arthritis % (Adults)",
    "ARTHRITIS_Crude95CI":      "Arthritis 95% CI",
    "BPHIGH_CrudePrev":         "High Blood Pressure % (Adults)",
    "BPHIGH_Crude95CI":         "High Blood Pressure 95% CI",
    "CANCER_CrudePrev":         "Cancer (Excl. Skin) % (Adults)",
    "CANCER_Crude95CI":         "Cancer 95% CI",
    "CASTHMA_CrudePrev":        "Current Asthma % (Adults)",
    "CASTHMA_Crude95CI":        "Current Asthma 95% CI",
    "CHD_CrudePrev":            "Coronary Heart Disease % (Adults)",
    "CHD_Crude95CI":            "Coronary Heart Disease 95% CI",
    "COPD_CrudePrev":           "COPD % (Adults)",
    "COPD_Crude95CI":           "COPD 95% CI",
    "DEPRESSION_CrudePrev":     "Depression % (Adults)",
    "DEPRESSION_Crude95CI":     "Depression 95% CI",
    "DIABETES_CrudePrev":       "Diabetes % (Adults)",
    "DIABETES_Crude95CI":       "Diabetes 95% CI",
    "HIGHCHOL_CrudePrev":       "High Cholesterol % (Adults)",
    "HIGHCHOL_Crude95CI":       "High Cholesterol 95% CI",
    "OBESITY_CrudePrev":        "Obesity % (Adults)",
    "OBESITY_Crude95CI":        "Obesity 95% CI",
    "STROKE_CrudePrev":         "Stroke % (Adults)",
    "STROKE_Crude95CI":         "Stroke 95% CI",
    "TEETHLOST_CrudePrev": "Teeth Lost % (Adults 65+)",
    "TEETHLOST_Crude95CI": "Teeth Lost 95% CI",

    # Preventive care
    "ACCESS2_CrudePrev":        "No Health Insurance % (Adults 18-64)",
    "ACCESS2_Crude95CI":        "No Health Insurance 95% CI",
    "BPMED_CrudePrev":          "Taking BP Medication % (Adults)",
    "BPMED_Crude95CI":          "Taking BP Medication 95% CI",
    "CHECKUP_CrudePrev":        "Annual Checkup % (Adults)",
    "CHECKUP_Crude95CI":        "Annual Checkup 95% CI",
    "CHOLSCREEN_CrudePrev":     "Cholesterol Screening % (Adults)",
    "CHOLSCREEN_Crude95CI":     "Cholesterol Screening 95% CI",
    "COLON_SCREEN_CrudePrev":   "Colorectal Cancer Screening % (Adults)",
    "COLON_SCREEN_Crude95CI":   "Colorectal Cancer Screening 95% CI",
    "DENTAL_CrudePrev":         "Dental Visit % (Adults)",
    "DENTAL_Crude95CI":         "Dental Visit 95% CI",
    "MAMMOUSE_CrudePrev":       "Mammography Use % (Women)",
    "MAMMOUSE_Crude95CI":       "Mammography Use 95% CI",

    # Health status
    "GHLTH_CrudePrev":          "Good or Better Health % (Adults)",
    "GHLTH_Crude95CI":          "Good or Better Health 95% CI",
    "MHLTH_CrudePrev":          "Poor Mental Health % (Adults)",
    "MHLTH_Crude95CI":          "Poor Mental Health 95% CI",
    "PHLTH_CrudePrev":          "Poor Physical Health % (Adults)",
    "PHLTH_Crude95CI":          "Poor Physical Health 95% CI",

    # Disability
    "DISABILITY_CrudePrev":     "Any Disability % (Adults)",
    "DISABILITY_Crude95CI":     "Any Disability 95% CI",
    "HEARING_CrudePrev":        "Hearing Disability % (Adults)",
    "HEARING_Crude95CI":        "Hearing Disability 95% CI",
    "VISION_CrudePrev":         "Vision Disability % (Adults)",
    "VISION_Crude95CI":         "Vision Disability 95% CI",
    "COGNITION_CrudePrev":      "Cognitive Disability % (Adults)",
    "COGNITION_Crude95CI":      "Cognitive Disability 95% CI",
    "MOBILITY_CrudePrev":       "Mobility Disability % (Adults)",
    "MOBILITY_Crude95CI":       "Mobility Disability 95% CI",
    "SELFCARE_CrudePrev":       "Self-Care Disability % (Adults)",
    "SELFCARE_Crude95CI":       "Self-Care Disability 95% CI",
    "INDEPLIVE_CrudePrev":      "Independent Living Disability % (Adults)",
    "INDEPLIVE_Crude95CI":      "Independent Living Disability 95% CI",

    # Social determinants
    "LONELINESS_CrudePrev":     "Social Isolation/Loneliness % (Adults)",
    "LONELINESS_Crude95CI":     "Social Isolation/Loneliness 95% CI",
    "FOODSTAMP_CrudePrev":      "SNAP/Food Stamp Use % (Adults)",
    "FOODSTAMP_Crude95CI":      "SNAP/Food Stamp Use 95% CI",
    "FOODINSECU_CrudePrev":     "Food Insecurity % (Adults)",
    "FOODINSECU_Crude95CI":     "Food Insecurity 95% CI",
    "HOUSINSECU_CrudePrev":     "Housing Insecurity % (Adults)",
    "HOUSINSECU_Crude95CI":     "Housing Insecurity 95% CI",
    "SHUTUTILITY_CrudePrev":    "Utility Shutoff Risk % (Adults)",
    "SHUTUTILITY_Crude95CI":    "Utility Shutoff Risk 95% CI",
    "LACKTRPT_CrudePrev":       "Lack of Transportation % (Adults)",
    "LACKTRPT_Crude95CI":       "Lack of Transportation 95% CI",
    "EMOTIONSPT_CrudePrev":     "Lack of Emotional Support % (Adults)",
    "EMOTIONSPT_Crude95CI":     "Lack of Emotional Support 95% CI",

    # Geolocation passthrough
    "geometry":              "Geolocation",
}



# ── Apply renames ─────────────────────────────────────────────────────────────
# Call this after step 8 in merge_all.py, passing in `out`.

def apply_renames(df):
    combined = {**ACS_RENAME, **PLACES_RENAME}

    missing = [k for k in combined if k not in df.columns]
    if missing:
        print(f"  ⚠ {len(missing)} rename keys not found in df (already renamed or absent):")
        for m in missing:
            print(f"      {m}")

    df = df.rename(columns=combined)
    return df


# ── Usage in merge_all.py ─────────────────────────────────────────────────────
# from rename_columns import apply_renames
#
# # After step 8:
# out = apply_renames(out)
#
# Then continue to step 9 (diagnostics) and step 10 (save).