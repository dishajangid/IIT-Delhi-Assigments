SELECT 
    a.subject_id,
    a.hadm_id,
    p.dod
FROM 
    hosp.patients p
JOIN 
    hosp.admissions a ON p.subject_id = a.subject_id
WHERE 
    p.dod IS NOT NULL
    AND a.admittime = (
        SELECT MIN(admittime) 
        FROM hosp.admissions 
        WHERE subject_id = a.subject_id
    )
ORDER BY 
    a.subject_id;

