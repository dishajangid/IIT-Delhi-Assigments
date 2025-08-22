WITH diagnoses_sets AS (
    SELECT 
        d.subject_id,
        d.hadm_id,
        STRING_AGG(DISTINCT d.icd_code, ',') AS diagnoses_set
    FROM 
        hosp.diagnoses_icd d
    GROUP BY 
        d.subject_id, d.hadm_id
),
medications_sets AS (
    SELECT 
        p.subject_id,
        p.hadm_id,
        STRING_AGG(DISTINCT p.drug, ',') AS medications_set
    FROM 
        hosp.prescriptions p
    GROUP BY 
        p.subject_id, p.hadm_id
)
SELECT 
    d.subject_id,
    COUNT(DISTINCT d.hadm_id) AS total_admissions,
    COUNT(DISTINCT d.diagnoses_set) AS num_distinct_diagnoses_set_count,
    COUNT(DISTINCT m.medications_set) AS num_distinct_medications_set_count
FROM 
    diagnoses_sets d
LEFT JOIN 
    medications_sets m ON d.subject_id = m.subject_id
GROUP BY 
    d.subject_id
HAVING 
    COUNT(DISTINCT d.diagnoses_set) >= 3 OR COUNT(DISTINCT m.medications_set) >= 3
ORDER BY 
    total_admissions DESC, num_distinct_diagnoses_set_count DESC, d.subject_id ASC;

