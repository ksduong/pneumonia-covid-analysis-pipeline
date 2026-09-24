-- ============================================================================
-- Cohort: COVID-19 positive and negative patients
-- Source: Montefiore OMOP CDM tables (MEASUREMENT, VISIT_OCCURRENCE)
-- ============================================================================

-- -----------------------------------------------------------------
-- COVID-19 positive cohort: first positive test per visit, followed
-- by a return visit >=30 days later (the observation window we
-- require to be able to detect new-onset outcomes).
-- -----------------------------------------------------------------
WITH covidTestConcept(concept_id) AS (
    VALUES (706163), (706170)       -- SARS-CoV-2 PCR test concepts
),
detectedConcept(concept_id) AS (
    VALUES (45877985), (9191)       -- "Detected" / positive result concepts
),
admissionConcept(concept_id) AS (
    VALUES (262), (9201), (32037)   -- Inpatient / ER / hospital encounter concepts
),
covids AS (
    -- earliest positive COVID test per visit
    SELECT
        person_id,
        m.visit_occurrence_id,
        MIN(m.measurement_datetime) AS measurement_datetime
    FROM measurement AS m
    WHERE m.measurement_concept_id IN (SELECT concept_id FROM covidTestConcept)
      AND m.value_as_concept_id IN (SELECT concept_id FROM detectedConcept)
    GROUP BY m.visit_occurrence_id, person_id
),
returns AS (
    -- most recent visit for each COVID+ patient, keeping only those who
    -- returned to the health system >=30 days after their positive test
    SELECT
        n.person_id,
        n.visit_occurrence_id,
        n.measurement_datetime,
        MAX(v.visit_start_datetime) AS return_visit_start
    FROM covids AS n
    JOIN visit_occurrence AS v USING (person_id)
    GROUP BY n.person_id, n.visit_occurrence_id, n.measurement_datetime
    HAVING DATE(MAX(v.visit_start_datetime)) > DATE(n.measurement_datetime, '+30 days')
)
SELECT * FROM returns;


-- -----------------------------------------------------------------
-- COVID-19 negative cohort: patients with a visit after 1 Mar 2020
-- who never appear in the positive-test set (covids, above) and are
-- not part of the separately identified influenza comparison cohort
-- -----------------------------------------------------------------
WITH covidTestConcept(concept_id) AS (
    VALUES (706163), (706170)
),
detectedConcept(concept_id) AS (
    VALUES (45877985), (9191)
),
covids AS (
    SELECT
        person_id,
        m.visit_occurrence_id,
        MIN(m.measurement_datetime) AS measurement_datetime
    FROM measurement AS m
    WHERE m.measurement_concept_id IN (SELECT concept_id FROM covidTestConcept)
      AND m.value_as_concept_id IN (SELECT concept_id FROM detectedConcept)
    GROUP BY m.visit_occurrence_id, person_id
)
SELECT
    person_id,
    MIN(visit_start_datetime) AS index_visit_start,
    visit_occurrence_id,
    visit_start_datetime,
    visit_end_datetime
FROM visit_occurrence
WHERE person_id NOT IN (SELECT person_id FROM covids)
  AND person_id NOT IN (SELECT person_id FROM other_covid_patients)  -- influenza cohort exclusion
  AND visit_start_datetime > '2020-03-01'
GROUP BY person_id;
