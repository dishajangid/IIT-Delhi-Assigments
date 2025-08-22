WITH admissions_data AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        d.icd_code,
        d.icd_version
    FROM
        (select * from hosp.admissions order by admittime limit 200) as a
    JOIN 
        hosp.diagnoses_icd d ON a.subject_id = d.subject_id AND a.hadm_id = d.hadm_id
    order by a.admittime
),
overlapping_admissions AS (
    SELECT
        ad1.subject_id AS subject_id1,
        ad2.subject_id AS subject_id2,
        ad1.hadm_id AS hadm_id1,
        ad2.hadm_id AS hadm_id2
    FROM
        admissions_data ad1
    JOIN 
        admissions_data ad2 ON ad1.subject_id < ad2.subject_id
    WHERE
        (ad1.admittime <= ad2.dischtime AND ad2.admittime < ad1.dischtime) 
        AND ad1.icd_code = ad2.icd_code
        AND ad1.icd_version = ad2.icd_version
)SELECT
    distinct subject_id1,
    subject_id2
FROM
    overlapping_admissions
ORDER BY
    subject_id1 ASC, subject_id2 ASC;

