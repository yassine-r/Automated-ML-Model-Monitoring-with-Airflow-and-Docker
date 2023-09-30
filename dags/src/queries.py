
CREATE_TABLE_ML_JOB = """
create table if not exists mljob (
    id serial primary key,
    job_id varchar(36) not null,
    job_type varchar(36),
    job_date date,
    stage varchar(36) not null,
    status varchar(36) not null,
    message text not null,
    created_at timestamp not null default now()
)
"""
LOG_ACTIVITY = """
    insert into mljob (
        job_id,
        job_type,
        job_date,
        stage,
        status,
        message
    ) values ('{job_id}', '{job_type}', '{job_date}', '{stage}', '{status}', '{message}')
"""


GET_JOB_STATUS = """
    select job_id, cast(job_date as varchar), stage, status, message, created_at 
    from mljob 
    where job_id = '{job_id}'
    order by created_at desc
    limit 1
"""

GET_JOB_DATE = """
    select job_date from mljob where job_id = '{job_id}'
    order by created_at desc
    limit 1
"""

GET_JOB_LOGS = """
    select job_id, cast(created_at as varchar), stage, status, message, created_at 
    from mljob 
    where job_id = '{job_id}' and message != '' and message is not null
    order by created_at desc
"""
GET_LATEST_TRAINING_JOB_ID = """
    select job_id from mljob
    where status = '{status}' and job_type = 'training' and stage = 'training'
    order by created_at desc
    limit 1
"""

GET_LATEST_DEPLOYED_JOB_ID = """
    select job_id from mljob
    where status = '{status}' and job_type = 'training' and stage = 'deploy'
    order by created_at desc
    limit 1
"""

GET_LATEST_JOB_ID = """
    select job_id from mljob
    where status = '{status}' and job_type = '{job_type}' and stage = '{stage}'
    order by created_at desc
    limit 1
"""