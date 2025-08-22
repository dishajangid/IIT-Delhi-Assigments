WITH distinct_procedures AS (
    SELECT 
        p.subject_id, 
        COUNT(DISTINCT p.icd_code) AS distinct_procedures_count
    FROM 
        hosp.procedures_icd p
    JOIN 
        hosp.diagnoses_icd d ON p.subject_id = d.subject_id AND p.hadm_id = d.hadm_id
    WHERE 
        d.icd_code LIKE 'T81%'  
    GROUP BY 
        p.subject_id
    HAVING 
        COUNT(DISTINCT p.icd_code) > 1  
),
transfer_counts AS (
    SELECT 
        t.subject_id, 
        COUNT(DISTINCT t.transfer_id) AS transfer_count
    FROM 
        hosp.transfers t 
    JOIN 
        hosp.admissions a ON t.subject_id = a.subject_id AND t.hadm_id = a.hadm_id 
    WHERE 
        t.subject_id IN (SELECT DISTINCT subject_id FROM distinct_procedures) 
    GROUP BY 
        t.subject_id
),
average_transfer_count AS (
    SELECT 
        AVG(transfer_count) AS avg_transfers
    FROM 
        transfer_counts
),
average_transfer_count_per_subject AS (
    SELECT 
        subject_id,
        AVG(transfer_count) AS avg_transfers_per_subject
    FROM 
        transfer_counts
    GROUP BY 
        subject_id
)
SELECT 
    dp.subject_id,
    dp.distinct_procedures_count,
    ats.avg_transfers_per_subject  
FROM 
    distinct_procedures dp
JOIN 
    transfer_counts tc ON dp.subject_id = tc.subject_id
JOIN 
    average_transfer_count at ON TRUE  
JOIN 
    average_transfer_count_per_subject ats ON dp.subject_id = ats.subject_id
WHERE 
    ats.avg_transfers_per_subject >= at.avg_transfers  
ORDER BY 
    ats.avg_transfers_per_subject DESC,  
    dp.distinct_procedures_count DESC,  
    dp.subject_id ASC;  

