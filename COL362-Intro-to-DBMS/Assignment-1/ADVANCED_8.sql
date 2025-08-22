WITH i10_admissions AS (
    SELECT d.subject_id, a.hadm_id AS i10_hadm_id, a.admittime AS i10_admittime
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'I10%' 
),
i50_admissions AS (
    SELECT d.subject_id, a.hadm_id AS i50_hadm_id, a.admittime AS i50_admittime
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'I50%'
),
admissions_between AS (
    SELECT i10.subject_id, i10.i10_hadm_id, i50.i50_hadm_id, i10.i10_admittime, i50.i50_admittime,
           a.hadm_id AS admission_id, a.admittime
    FROM i10_admissions i10
    JOIN i50_admissions i50 ON i10.subject_id = i50.subject_id
    JOIN hosp.admissions a ON a.subject_id = i10.subject_id
    WHERE a.admittime > i10.i10_admittime
      AND a.admittime < i50.i50_admittime  
),
valid_admissions AS (
    SELECT subject_id, 
           MIN(admittime) AS first_admittime,
           MAX(admittime) AS last_admittime, 
           COUNT(*) AS admission_count
    FROM admissions_between
    GROUP BY subject_id
    HAVING COUNT(*) > 2 
),
distinct_drugs AS (
    SELECT a.subject_id, a.hadm_id AS admission_id, p.drug
    FROM valid_admissions va
    JOIN hosp.prescriptions p ON va.subject_id = p.subject_id
    JOIN hosp.admissions a ON p.hadm_id = a.hadm_id
    WHERE a.admittime > va.first_admittime  
      AND a.admittime < va.last_admittime   
)
SELECT DISTINCT subject_id, admission_id, drug
FROM distinct_drugs
ORDER BY subject_id ASC, admission_id ASC, drug ASC;
