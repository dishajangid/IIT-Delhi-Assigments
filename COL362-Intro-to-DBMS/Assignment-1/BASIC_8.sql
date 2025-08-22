SELECT pharmacy_id
FROM hosp.pharmacy
WHERE pharmacy_id NOT IN (SELECT DISTINCT pharmacy_id FROM hosp.prescriptions)
ORDER BY pharmacy_id;

