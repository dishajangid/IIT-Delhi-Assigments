WITH drug_prescriptions AS (
    SELECT subject_id, hadm_id, LOWER(drug) AS drug
    FROM hosp.prescriptions
    WHERE LOWER(drug) IN ('amlodipine', 'lisinopril')
),
drug_combination AS (
    SELECT subject_id, hadm_id,
           CASE 
               WHEN COUNT(DISTINCT drug) = 2 THEN 'both'  
               WHEN COUNT(DISTINCT drug) = 1 AND MAX(drug) = 'amlodipine' THEN 'amlodipine'  
               WHEN COUNT(DISTINCT drug) = 1 AND MAX(drug) = 'lisinopril' THEN 'lisinopril'  
           END AS drug
    FROM drug_prescriptions
    GROUP BY subject_id, hadm_id
),
service_path AS (
    SELECT s.subject_id, s.hadm_id, s.transfertime, s.prev_service, s.curr_service,
           ROW_NUMBER() OVER (PARTITION BY s.subject_id, s.hadm_id ORDER BY s.transfertime) AS service_order
    FROM hosp.services s
    WHERE (s.prev_service IS NULL AND s.curr_service IS NOT NULL)  
    OR (s.prev_service IS NOT NULL AND s.curr_service IS NOT NULL)  
)
SELECT dp.subject_id,
       dp.hadm_id,
       dp.drug,
       COALESCE(ARRAY_AGG(sp.curr_service ORDER BY sp.service_order), ARRAY[]::text[]) AS services
FROM drug_combination dp
LEFT JOIN service_path sp
    ON dp.subject_id = sp.subject_id
    AND dp.hadm_id = sp.hadm_id
GROUP BY dp.subject_id, dp.hadm_id, dp.drug 
ORDER BY dp.subject_id, dp.hadm_id;

