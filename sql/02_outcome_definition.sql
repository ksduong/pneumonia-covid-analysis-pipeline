-- ============================================================================
-- Outcome definition: new-onset pneumonia
-- Patient counts as new onset if their earliest pneumonia diagnosis starts 
-- >=30 days after COVID index date and they had no pneumonia diagnosis before it.
-- Concept IDs below are the OMOP standard concepts for pneumonia.
-- ============================================================================

WITH pneumonia_concepts(concept_id) AS (
    VALUES
        (252351), (252655), (252949), (253235), (253790), (254066), (254677),
        (255084), (255848), (256722), (256723), (257315), (258180), (258785),
        (259852), (259992), (260430), (260754), (261324), (261326), (436145),
        (439857), (440431), (443410), (3661408), (4025165), (4044215), (4070540),
        (4071610), (4071611), (4110510), (4112820), (4133224), (4174309),
        (4195014), (4231983), (4310964), (4311555), (4341520), (36676238),
        (36714118), (40479642), (40482061), (40489912), (45763749), (45533545),
        (45557620), (35207937), (35207938), (35207939), (35207940), (35207941),
        (35207942), (35207943), (35207944), (35207945), (35207947), (35207948),
        (35207949), (35207950), (35207951), (35207952), (35207953), (35207956)
),
first_pneumonia_dx AS (
    -- earliest pneumonia diagnosis per patient
    SELECT
        person_id,
        condition_concept_id,
        MIN(condition_start_datetime) AS condition_start_datetime
    FROM condition_occurrence
    WHERE condition_concept_id IN (SELECT concept_id FROM pneumonia_concepts)
    GROUP BY person_id, condition_concept_id
)
SELECT
    r.person_id,
    r.measurement_datetime AS covid_index_date,
    p.condition_start_datetime AS pneumonia_date,
    CASE
        WHEN p.condition_start_datetime IS NULL THEN 0
        WHEN DATE(p.condition_start_datetime) >= DATE(r.measurement_datetime, '+30 days') THEN 1
        ELSE 0  -- prior pneumonia (excluded from the new-onset outcome)
    END AS new_onset_pneumonia
FROM returns AS r -- from 01_cohort_identification.sql
LEFT JOIN first_pneumonia_dx AS p USING (person_id);