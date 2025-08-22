WITH RECURSIVE admissions_data AS (
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
),
bfs_traversal AS (
    SELECT
        subject_id1 AS current_subject_id,
        ARRAY[subject_id1] AS path  
    FROM
        overlapping_admissions
    WHERE
        subject_id1 = 10038081 
    
    UNION ALL
    SELECT
        oa.subject_id2 AS current_subject_id,
        path || oa.subject_id2 AS path  
    FROM
        overlapping_admissions oa
    JOIN
        bfs_traversal bfs ON oa.subject_id1 = bfs.current_subject_id 
    WHERE
        NOT oa.subject_id2 = ANY(bfs.path) 
)
SELECT
    (COUNT(DISTINCT current_subject_id) - 1) AS count
FROM
    bfs_traversal;

