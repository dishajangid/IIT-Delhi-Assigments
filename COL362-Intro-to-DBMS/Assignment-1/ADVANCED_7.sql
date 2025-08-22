WITH e10_e11_admissions AS (
    SELECT d.subject_id, a.hadm_id AS admission_id, d.icd_code
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%'  
),
n18_admissions AS (
    SELECT d.subject_id, a.hadm_id AS admission_id, d.icd_code
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'N18%' 
),
all_admissions AS (
    SELECT e.subject_id, e.admission_id, e.icd_code, 'diagnoses' AS diagnoses_or_procedure
    FROM e10_e11_admissions e
    JOIN n18_admissions n
        ON e.subject_id = n.subject_id
        AND n.admission_id >= e.admission_id 
    UNION
    SELECT n.subject_id, n.admission_id, n.icd_code, 'diagnoses' AS diagnoses_or_procedure
    FROM n18_admissions n
    WHERE EXISTS (
        SELECT 1
        FROM e10_e11_admissions e
        WHERE e.subject_id = n.subject_id
        AND e.admission_id <= n.admission_id
    )
),
procedure_admissions AS (
    SELECT p.subject_id, p.hadm_id AS admission_id, p.icd_code, 'procedures' AS diagnoses_or_procedure
    FROM hosp.procedures_icd p
    JOIN hosp.admissions a ON p.subject_id = a.subject_id AND p.hadm_id = a.hadm_id
    WHERE EXISTS (
        SELECT 1
        FROM all_admissions e
        WHERE e.subject_id = p.subject_id AND e.admission_id = p.hadm_id
    )
),
final_result AS (
    SELECT subject_id, admission_id, diagnoses_or_procedure, icd_code
    FROM all_admissions
    UNION
    SELECT subject_id, admission_id, diagnoses_or_procedure, icd_code
    FROM procedure_admissions
)
SELECT DISTINCT subject_id, admission_id, diagnoses_or_procedure, icd_code
FROM final_result
ORDER BY subject_id ASC, admission_id ASC, icd_code ASC, diagnoses_or_procedure ASC;

