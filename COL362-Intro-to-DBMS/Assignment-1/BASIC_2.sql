select count(distinct a.hadm_id) as count, EXTRACT(year from admittime::date) as year from hosp.admissions a group by year order by count desc, year limit 5;
