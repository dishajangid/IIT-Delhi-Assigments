WITH admissions_data AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime
    FROM
        hosp.admissions a
    ORDER BY
        a.admittime 
    LIMIT 200 
),
overlapping_admissions AS (
    SELECT
        ad1.subject_id AS subject_id1,
        ad2.subject_id AS subject_id2
    FROM
        admissions_data ad1
    JOIN
        admissions_data ad2
    ON
        ad1.subject_id != ad2.subject_id  
        AND ad1.admittime <= ad2.dischtime  
        AND ad1.dischtime > ad2.admittime  
)
SELECT 
    CASE 
        WHEN EXISTS (
            SELECT 1
            FROM overlapping_admissions
            WHERE (subject_id1 = 10006580 AND subject_id2 = 10003400)
               OR (subject_id1 = 10003400 AND subject_id2 = 10006580)
        ) THEN 1
        ELSE 0
    END AS path_exists;

