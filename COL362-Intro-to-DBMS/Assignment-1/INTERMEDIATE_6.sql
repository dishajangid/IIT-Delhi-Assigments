WITH prescribed_medications AS (
    SELECT 
        o.subject_id
    FROM 
        hosp.omr o
    JOIN 
        hosp.prescriptions p1 ON p1.subject_id = o.subject_id
    JOIN 
        hosp.prescriptions p2 ON p2.subject_id = o.subject_id
    WHERE 
        p1.drug = 'OxyCODONE (Immediate Release)'  
        AND p2.drug = 'Insulin'                   
    GROUP BY 
        o.subject_id
)
SELECT 
    ROUND(AVG(o.result_value::numeric), 10) AS avg_BMI  
FROM 
    hosp.omr o
JOIN 
    prescribed_medications pm ON o.subject_id = pm.subject_id 
WHERE 
    o.result_name LIKE '%BMI%';  
