SELECT 
    p.subject_id, 
    a.hadm_id AS latest_hadm_id, 
    p.dod
FROM 
    hosp.patients p
JOIN 
    hosp.admissions a ON p.subject_id = a.subject_id
WHERE 
    p.dod IS NOT NULL
AND 
    a.admittime = (
        SELECT MAX(admittime)
        FROM hosp.admissions
        WHERE subject_id = p.subject_id
    )
ORDER BY 
    p.subject_id;

