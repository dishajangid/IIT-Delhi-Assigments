SELECT 
    a.subject_id,
    a.hadm_id,
    p.count AS count_procedures,
    COUNT(DISTINCT di.icd_code || di.icd_version) AS count_diagnoses
FROM 
    hosp.admissions a
LEFT JOIN 
    (select p.hadm_id, count(*) as count from hosp.procedures_icd p group by p.hadm_id) as p on a.hadm_id = p.hadm_id
LEFT JOIN 
    hosp.diagnoses_icd di ON a.hadm_id = di.hadm_id
WHERE 
    a.admission_type = 'URGENT'
    AND a.hospital_expire_flag = 1
GROUP BY 
    a.subject_id, a.hadm_id, p.count
ORDER BY 
    a.subject_id, a.hadm_id, count_procedures DESC, count_diagnoses DESC;

