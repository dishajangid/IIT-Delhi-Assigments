WITH fluid_balance AS (
    SELECT
        i.subject_id,
        i.stay_id,
        SUM(CASE WHEN di.linksto = 'inputevents' THEN i.amount ELSE 0 END) AS total_input_ml,
        SUM(CASE WHEN di.linksto = 'outputevents' THEN o.value ELSE 0 END) AS total_output_ml
    FROM 
        icu.inputevents i
    JOIN 
        icu.d_items di ON i.itemid = di.itemid
    LEFT JOIN 
        icu.outputevents o ON i.subject_id = o.subject_id AND i.hadm_id = o.hadm_id AND i.stay_id = o.stay_id
    JOIN 
        icu.icustays s ON i.subject_id = s.subject_id AND i.hadm_id = s.hadm_id
    WHERE 
        i.amountuom = 'ml' AND o.valueuom = 'ml'
    GROUP BY 
        i.subject_id, i.stay_id
    HAVING 
        ABS(SUM(CASE WHEN di.linksto = 'inputevents' THEN i.amount ELSE 0 END) - 
            SUM(CASE WHEN di.linksto = 'outputevents' THEN o.value ELSE 0 END)) > 2000
)

SELECT 
    DISTINCT i.subject_id, 
    i.stay_id,
    i.itemid,
    CASE 
        WHEN di.linksto = 'inputevents' THEN 'input' 
        WHEN di.linksto = 'outputevents' THEN 'output'
        ELSE 'unknown'  
    END AS input_or_output,
    di.abbreviation AS description
FROM 
    icu.inputevents i
JOIN 
    icu.icustays s ON i.subject_id = s.subject_id AND i.hadm_id = s.hadm_id
JOIN 
    icu.d_items di ON i.itemid = di.itemid
JOIN 
    fluid_balance fb ON i.subject_id = fb.subject_id AND i.stay_id = fb.stay_id
WHERE 
    i.amountuom = 'ml' 
    
UNION ALL

SELECT 
    DISTINCT o.subject_id, 
    o.stay_id,
    o.itemid,
    CASE 
        WHEN di.linksto = 'inputevents' THEN 'input' 
        WHEN di.linksto = 'outputevents' THEN 'output'
        ELSE 'unknown'  
    END AS input_or_output,
    di.abbreviation AS description
FROM 
    icu.outputevents o
JOIN 
    icu.icustays s ON o.subject_id = s.subject_id AND o.hadm_id = s.hadm_id
JOIN 
    icu.d_items di ON o.itemid = di.itemid
JOIN 
    fluid_balance fb ON o.subject_id = fb.subject_id AND o.stay_id = fb.stay_id
WHERE 
    o.valueuom = 'ml'  
    
ORDER BY 
    subject_id ASC, 
    stay_id ASC, 
    itemid ASC, 
    input_or_output ASC;

