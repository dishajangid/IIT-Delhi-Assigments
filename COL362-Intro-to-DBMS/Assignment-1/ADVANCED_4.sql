WITH transfer_chain_lengths AS (
    SELECT 
        t.subject_id, 
        t.hadm_id, 
        COUNT(DISTINCT t.transfer_id) AS transfer_chain_length,
        ARRAY_AGG(t.transfer_id ORDER BY t.intime) AS transfer_ids
    FROM 
        hosp.transfers t
    GROUP BY 
        t.subject_id, t.hadm_id
),
max_transfer_chain AS (
    SELECT 
        subject_id
    FROM 
        transfer_chain_lengths
    WHERE 
        transfer_chain_length = (SELECT MAX(transfer_chain_length) FROM transfer_chain_lengths)
)
SELECT 
    tcl.subject_id, 
    tcl.hadm_id, 
    tcl.transfer_ids
FROM 
    transfer_chain_lengths tcl
JOIN 
    max_transfer_chain mtc ON tcl.subject_id = mtc.subject_id
WHERE 
    tcl.hadm_id IS NOT NULL
ORDER BY 
    array_length(tcl.transfer_ids, 1) ASC, tcl.hadm_id ASC, tcl.subject_id ASC;

