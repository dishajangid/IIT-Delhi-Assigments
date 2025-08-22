WITH first_admissions AS (
    SELECT 
        a.subject_id,
        (SELECT a1.hadm_id
         FROM hosp.admissions a1
         WHERE a1.subject_id = a.subject_id
         ORDER BY a1.admittime
         LIMIT 1) AS first_hadm_id
    FROM 
        hosp.admissions a
    JOIN hosp.diagnoses_icd di ON a.hadm_id = di.hadm_id
    JOIN hosp.d_icd_diagnoses d ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version
    WHERE LOWER(d.long_title) LIKE '%kidney%'
    GROUP BY 
        a.subject_id
),
patients_with_multiple_admissions AS (
    SELECT 
        subject_id
    FROM 
        hosp.admissions
    GROUP BY 
        subject_id
    HAVING COUNT(distinct hadm_id) > 1
)
select subject_id
from first_admissions
intersect
select subject_id
from patients_with_multiple_admissions 
order by subject_id;
