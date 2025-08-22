WITH FirstAdmissionDiagnoses AS (
    SELECT 
        a.subject_id,
        (SELECT a1.hadm_id
         FROM hosp.admissions a1
         WHERE a1.subject_id = a.subject_id
         ORDER BY a1.admittime
         LIMIT 1) AS first_hadm_id
    FROM 
        hosp.admissions a
    JOIN 
        hosp.diagnoses_icd di ON a.hadm_id = di.hadm_id
    JOIN 
        hosp.d_icd_diagnoses d ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version
    GROUP BY 
        a.subject_id
),
MostRecentAdmissionDiagnoses AS (
    SELECT 
        a.subject_id,
        (SELECT a1.hadm_id
         FROM hosp.admissions a1
         WHERE a1.subject_id = a.subject_id
         ORDER BY a1.admittime DESC
         LIMIT 1) AS most_recent_hadm_id
    FROM 
        hosp.admissions a
    JOIN 
        hosp.diagnoses_icd di ON a.hadm_id = di.hadm_id
    JOIN 
        hosp.d_icd_diagnoses d ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version 
    GROUP BY 
        a.subject_id
),
SameDiagnosisPatients AS (
    SELECT 
        f.subject_id
    FROM 
        FirstAdmissionDiagnoses f
    JOIN 
        MostRecentAdmissionDiagnoses m ON f.subject_id = m.subject_id
    JOIN 
        hosp.diagnoses_icd di_first ON f.first_hadm_id = di_first.hadm_id
    JOIN 
        hosp.diagnoses_icd di_recent ON m.most_recent_hadm_id = di_recent.hadm_id
    WHERE 
        di_first.icd_code = di_recent.icd_code  -- Check if the ICD code is the same
)
SELECT 
    p.gender,
    ROUND(
        (COUNT(DISTINCT s.subject_id) * 100.0) / 
        (SELECT COUNT(DISTINCT subject_id) FROM SameDiagnosisPatients), 2
    ) AS percentage
FROM 
    SameDiagnosisPatients s
JOIN 
    hosp.patients p ON s.subject_id = p.subject_id
GROUP BY 
    p.gender
ORDER BY 
    percentage DESC;

