WITH count_p AS (
    SELECT 
        cg.caregiver_id, 
        COUNT(DISTINCT p.hadm_id) AS procedureevents_count
    FROM 
        icu.caregiver cg 
    LEFT JOIN 
        icu.procedureevents p ON p.caregiver_id = cg.caregiver_id
    GROUP BY 
        cg.caregiver_id
),
count_c AS (
    SELECT 
        cg.caregiver_id, 
        COUNT(DISTINCT c.hadm_id) AS chartevents_count
    FROM 
        icu.caregiver cg 
    LEFT JOIN 
        icu.chartevents c ON c.caregiver_id = cg.caregiver_id
    GROUP BY 
        cg.caregiver_id
),
count_dt AS (
    SELECT 
        cg.caregiver_id, 
        COUNT(DISTINCT t.hadm_id) AS datetimeevents_count
    FROM 
        icu.caregiver cg 
    LEFT JOIN 
        icu.datetimeevents t ON t.caregiver_id = cg.caregiver_id
    GROUP BY 
        cg.caregiver_id
)

SELECT 
    cg.caregiver_id,
    COALESCE(count_p.procedureevents_count, 0) AS procedureevents_count,
    COALESCE(count_c.chartevents_count, 0) AS chartevents_count,
    COALESCE(count_dt.datetimeevents_count, 0) AS datetimeevents_count
FROM 
    icu.caregiver cg
LEFT JOIN 
    count_p ON cg.caregiver_id = count_p.caregiver_id
LEFT JOIN 
    count_c ON cg.caregiver_id = count_c.caregiver_id
LEFT JOIN 
    count_dt ON cg.caregiver_id = count_dt.caregiver_id
ORDER BY 
    cg.caregiver_id;

