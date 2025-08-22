SELECT 
    (EXTRACT(day FROM AVG(dischtime::timestamp - admittime::timestamp)) + 
     FLOOR(EXTRACT(hour FROM AVG(dischtime::timestamp - admittime::timestamp)) / 24)) || ' days ' ||
    LPAD((EXTRACT(hour FROM AVG(dischtime::timestamp - admittime::timestamp)) % 24)::TEXT, 2, '0') || ':' || 
    LPAD((EXTRACT(minute FROM AVG(dischtime::timestamp - admittime::timestamp)))::TEXT, 2, '0') || ':' || 
    LPAD((EXTRACT(second FROM AVG(dischtime::timestamp - admittime::timestamp)))::TEXT, 2, '0') AS avg_duration
FROM 
    hosp.admissions a
JOIN 
    hosp.diagnoses_icd di ON a.hadm_id = di.hadm_id
WHERE 
    di.icd_code = '4019' 
    AND di.icd_version = 9;

