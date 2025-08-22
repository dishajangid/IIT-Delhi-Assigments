SELECT 
    a.subject_id, 
    COUNT(distinct a.hadm_id) AS count_admissions, 
    EXTRACT(YEAR FROM TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS')) AS year
FROM 
    hosp.admissions a
JOIN 
    hosp.diagnoses_icd di ON a.subject_id = di.subject_id AND a.hadm_id = di.hadm_id  
JOIN 
    hosp.d_icd_diagnoses d ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version 
WHERE 
    d.long_title ILIKE '%infection%' 
GROUP BY 
    a.subject_id, year
HAVING 
    COUNT(distinct a.hadm_id) > 1
ORDER BY 
    year ASC, count_admissions DESC, a.subject_id;

