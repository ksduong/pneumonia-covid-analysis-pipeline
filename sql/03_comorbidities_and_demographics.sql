-- ============================================================================
-- Comorbidity extraction and demographics concept mapping
-- ============================================================================

-- -----------------------------------------------------------------
-- Generic comorbidity query shape (repeated per condition, using
-- different concept_id lists -- see `comorbidity_concepts.py`.
-- -----------------------------------------------------------------
SELECT
    person_id,
    condition_concept_id,
    condition_start_datetime
FROM condition_occurrence
WHERE condition_concept_id IN (/* per-condition concept list */);

-- tobacco use recorded in OBSERVATION
SELECT
    person_id,
    observation_concept_id,
    observation_datetime
FROM observation
WHERE observation_concept_id = 4041306;


-- -----------------------------------------------------------------
-- Demographics
-- -----------------------------------------------------------------
SELECT *
FROM person
WHERE person_id IN (/* cohort person_ids */)
ORDER BY person_id;

-- concept mappings used to decode PERSON table fields:

-- sex / gender_concept_id
--   8532 -> female
--   8507 -> male

-- ethnicity_concept_id
--   0        -> no matching concept (*treated as not hispanic/latino)
--   38003563 -> Hispanic or Latino
--   38003564 -> Not Hispanic or Latino

-- race_concept_id
--   0        -> no matching concept (*treated as other/unknown)
--   8515     -> Asian
--   8516     -> Black or African American
--   8527     -> White
--   8557     -> Native Hawaiian or Other Pacific Islander
--   4218674  -> Unknown racial group
--   38003613 -> Other Pacific Islander
