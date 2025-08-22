SELECT 
    a.subject_id, 
    (EXTRACT(day FROM AVG((a.dischtime::timestamp - a.admittime::timestamp))) + 
     FLOOR(EXTRACT(hour FROM AVG((a.dischtime::timestamp - a.admittime::timestamp))) / 24)) || ' days ' ||
    LPAD((EXTRACT(hour FROM AVG((a.dischtime::timestamp - a.admittime::timestamp))) % 24)::TEXT, 2, '0') || ':' || 
    LPAD((EXTRACT(minute FROM AVG((a.dischtime::timestamp - a.admittime::timestamp))) )::TEXT, 2, '0') || ':' || 
    LPAD((EXTRACT(second FROM AVG((a.dischtime::timestamp - a.admittime::timestamp))) )::TEXT, 2, '0') AS avg_duration
FROM 
    hosp.admissions a
GROUP BY 
    a.subject_id
ORDER BY 
    a.subject_id;

