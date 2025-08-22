select e.subject_id, e.pharmacy_id
from hosp.prescriptions e where e.pharmacy_id is NOT NULL 
group by e.subject_id, e.pharmacy_id 
having count(e.hadm_id) > 1 
order by e.subject_id, e.pharmacy_id;

