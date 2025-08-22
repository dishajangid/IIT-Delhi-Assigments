select enter_provider_id, count(distinct medication) as count 
from hosp.emar
where enter_provider_id is NOT NULL  
group by enter_provider_id 
ORDER BY count desc;
