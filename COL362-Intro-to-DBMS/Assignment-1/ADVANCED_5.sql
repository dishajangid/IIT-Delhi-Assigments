WITH procedures_with_medications AS (
    SELECT
        pe.subject_id,
        pe.hadm_id,
        pe.chartdate AS procedure_starttime,
        p.drug,
        p.starttime AS medication_time,
        pi.icd_code
    FROM
        hosp.procedures_icd pe
    JOIN
        hosp.prescriptions p ON pe.subject_id = p.subject_id AND pe.hadm_id = p.hadm_id
    JOIN
        hosp.procedures_icd pi ON pi.subject_id = pe.subject_id AND pi.hadm_id = pe.hadm_id
    WHERE
        (pi.icd_code LIKE '0%' OR pi.icd_code LIKE '1%' OR pi.icd_code LIKE '2%')
        AND (DATE(pe.chartdate) = DATE(p.starttime) OR DATE(pe.chartdate) + INTERVAL '1 day' = DATE(p.starttime))
),
distinct_medications AS (
    SELECT
        subject_id,
        hadm_id,
        COUNT(DISTINCT drug) AS distinct_drugs_count
    FROM
        hosp.prescriptions
    GROUP BY
        subject_id, hadm_id
    HAVING
        COUNT(DISTINCT drug) > 1
),
procedure_diagnoses AS (
    SELECT
        p.subject_id,
        p.hadm_id,
        COUNT(DISTINCT pi.icd_code) AS distinct_procedures
    FROM
        hosp.procedures_icd p
    JOIN
        hosp.procedures_icd pi ON pi.subject_id = p.subject_id AND pi.hadm_id = p.hadm_id
    GROUP BY
        p.subject_id, p.hadm_id
),
distinct_diagnoses AS (
    SELECT
        d.subject_id,
        d.hadm_id,
        COUNT(DISTINCT d.icd_code) AS distinct_diagnoses
    FROM
        hosp.diagnoses_icd d
    GROUP BY
        d.subject_id, d.hadm_id
),
final_results AS (
    SELECT
        p.subject_id,
        p.hadm_id,
        d.distinct_diagnoses,
        pr.distinct_procedures,
        age(MAX(CAST(p.medication_time AS timestamp)), MIN(CAST(p.procedure_starttime AS timestamp))) AS time_gap
    FROM
        procedures_with_medications p
    JOIN
        distinct_medications m ON p.subject_id = m.subject_id AND p.hadm_id = m.hadm_id
    JOIN
        procedure_diagnoses pr ON p.subject_id = pr.subject_id AND p.hadm_id = pr.hadm_id
    JOIN
        distinct_diagnoses d ON p.subject_id = d.subject_id AND p.hadm_id = d.hadm_id
    GROUP BY
        p.subject_id, p.hadm_id, d.distinct_diagnoses, pr.distinct_procedures
)
SELECT
    subject_id,
    hadm_id,
    distinct_diagnoses,
    distinct_procedures,
    -- Format the time gap as "YYYY-MM-DD HH:MM:SS"
    TO_CHAR(time_gap, 'YYYY-MM-DD HH24:MI:SS') AS time_gap
FROM
    final_results
ORDER BY
    distinct_diagnoses DESC, distinct_procedures DESC, time_gap ASC, subject_id ASC, hadm_id ASC;

