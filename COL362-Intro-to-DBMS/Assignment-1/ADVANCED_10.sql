WITH first_admission AS (
    SELECT 
        d.subject_id, 
        a.hadm_id AS first_hadm_id, 
        a.dischtime AS first_dischtime, 
        a.admittime AS first_admittime
    FROM 
        hosp.diagnoses_icd d
    JOIN 
        hosp.admissions a ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
    WHERE 
        d.icd_code LIKE 'I2%'  
),
second_admission AS (
    SELECT DISTINCT ON (a.subject_id) 
        a.subject_id, 
        a.hadm_id AS second_hadm_id, 
        a.admittime AS second_admittime, 
        a.dischtime AS second_dischtime,
        fa.first_dischtime
    FROM 
        hosp.admissions a
    JOIN 
        first_admission fa ON a.subject_id = fa.subject_id
    WHERE 
        a.admittime::timestamp > fa.first_dischtime::timestamp  
        AND a.admittime::timestamp <= (fa.first_dischtime::timestamp + INTERVAL '180 days') 
    ORDER BY 
        a.subject_id, a.admittime DESC  -- Ensure we get the most recent admission
),
service_path AS (
    SELECT 
        s.subject_id, 
        s.hadm_id AS second_hadm_id, 
        s.transfertime, 
        s.curr_service,
        ROW_NUMBER() OVER (PARTITION BY s.subject_id, s.hadm_id ORDER BY s.transfertime) AS service_order
    FROM 
        hosp.services s
    JOIN 
        second_admission sa ON s.subject_id = sa.subject_id AND s.hadm_id = sa.second_hadm_id
),
aggregated_services AS (
    SELECT 
        sa.subject_id, 
        sa.second_hadm_id,
        COALESCE(ARRAY_AGG(sp.curr_service ORDER BY sp.service_order), ARRAY[]::text[]) AS services
    FROM 
        second_admission sa
    LEFT JOIN 
        service_path sp ON sa.subject_id = sp.subject_id AND sa.second_hadm_id = sp.second_hadm_id
    GROUP BY 
        sa.subject_id, sa.second_hadm_id
)
SELECT 
    sa.subject_id, 
    sa.second_hadm_id,
    TO_CHAR(AGE(sa.second_admittime::timestamp, sa.first_dischtime::timestamp), 'YYYY-MM-DD HH24:MI:SS') AS time_gap,
    ag.services
FROM 
    second_admission sa
JOIN 
    aggregated_services ag ON sa.subject_id = ag.subject_id AND sa.second_hadm_id = ag.second_hadm_id
ORDER BY 
    array_length(ag.services, 1) DESC,  
    time_gap DESC,  
    sa.subject_id ASC, 
    sa.second_hadm_id ASC;

