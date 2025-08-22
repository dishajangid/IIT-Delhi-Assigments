select hosp.patients.subject_id, count(icu.icustays.hadm_id) as count 
from hosp.patients, icu.icustays 
where hosp.patients.subject_id = icu.icustays.subject_id 
group by patients.subject_id 
order by count, patients.subject_id;
