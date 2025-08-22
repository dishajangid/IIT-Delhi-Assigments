SELECT 
    COUNT(DISTINCT a.hadm_id) AS count
FROM 
    hosp.admissions a
JOIN 
    hosp.emar_detail e ON a.subject_id = e.subject_id
WHERE 
    e.reason_for_no_barcode = 'Barcode Damaged' 
    AND a.marital_status <> 'MARRIED';

