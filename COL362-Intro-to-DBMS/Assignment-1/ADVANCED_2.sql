SELECT 
    m.subject_id, 
    m.hadm_id, 
    COUNT(DISTINCT m.micro_specimen_id) AS resistant_antibiotic_count,
    ROUND(COALESCE(
        EXTRACT(EPOCH FROM (TO_TIMESTAMP(i.outtime, 'YYYY-MM-DD HH24:MI:SS') - TO_TIMESTAMP(i.intime, 'YYYY-MM-DD HH24:MI:SS'))) / 3600, 
        0
    ), 2) AS icu_length_of_stay_hours,
    CASE 
        WHEN a.discharge_location = 'DIED' THEN 1
        ELSE 0
    END AS died_in_hospital
FROM 
    hosp.microbiologyevents m
LEFT JOIN 
    icu.icustays i ON m.subject_id = i.subject_id AND m.hadm_id = i.hadm_id
JOIN 
    hosp.admissions a ON m.subject_id = a.subject_id AND m.hadm_id = a.hadm_id
WHERE 
    m.interpretation = 'R'
GROUP BY 
    m.subject_id, m.hadm_id, i.intime, i.outtime, a.discharge_location
HAVING 
    COUNT(DISTINCT m.micro_specimen_id) >= 2
ORDER BY 
    died_in_hospital DESC, 
    resistant_antibiotic_count DESC, 
    icu_length_of_stay_hours DESC, 
    m.subject_id ASC, 
    m.hadm_id ASC;

