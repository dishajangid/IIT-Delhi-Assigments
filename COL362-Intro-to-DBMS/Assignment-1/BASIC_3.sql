SELECT hadm_id, gender, (dischtime::timestamp - admittime::timestamp) AS duration
FROM hosp.admissions                            
JOIN hosp.patients ON hosp.patients.subject_id = hosp.admissions.subject_id where dischtime is NOT NULL order by duration, hadm_id;

