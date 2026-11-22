# Which workload do first jobs run?

Read-only admin query for the question "what do new customers launch first?".

## Where the answer is recorded

Each job's execution spec (`jobs.spec`) holds:

| Field | Set when | Meaning |
|---|---|---|
| `launch_template_id` | the launch form started from a quick-start template and the image is still that template's image (since 2026-11-22) | an `id` from `GET /api/images/templates` (`auth/security.py` `IMAGE_TEMPLATES`) |
| `source_template_id` | the launch named a saved image (`template_image_id`) | a `user_images.image_id` |
| `image` | always | the image the job ran |

Jobs submitted before 2026-11-22 have no `launch_template_id`, so they group
by image. A template launch from before then shows as `image:<template image>`.
`image:` with nothing after it is a job that recorded no image.

## First job per customer, by workload

```sql
WITH first_job AS (
  SELECT DISTINCT ON (j.owner_id)
         j.owner_id,
         j.job_id,
         to_timestamp(j.submitted_at) AS submitted,
         j.status,
         j.spec->>'launch_template_id' AS launch_template_id,
         NULLIF(j.spec->>'source_template_id', '') AS saved_image_id,
         COALESCE(j.spec->>'image', j.payload->>'image') AS image
    FROM jobs j
   WHERE j.owner_id IS NOT NULL AND j.owner_id <> ''
     AND COALESCE(j.spec->>'job_type', '') NOT IN ('serverless_worker')
     AND j.spec->>'hpc_cluster_id' IS NULL
   ORDER BY j.owner_id, j.submitted_at, j.job_id
)
SELECT CASE
         WHEN launch_template_id IS NOT NULL THEN 'template:' || launch_template_id
         WHEN saved_image_id IS NOT NULL THEN 'saved image'
         ELSE 'image:' || split_part(image, '@', 1)
       END AS workload,
       count(*) AS customers,
       count(*) FILTER (WHERE status = 'completed') AS completed,
       min(submitted) AS first_seen,
       max(submitted) AS last_seen
  FROM first_job
 GROUP BY 1
 ORDER BY customers DESC;
```

Run it against a read replica or with a read-only role. It lists image names,
which are customer data: keep the output out of tickets, chats and analytics.

## The same question in PostHog

`gpu_instance_launched` carries `template_id`, `workload_kind`
(`template` / `saved_image` / `custom_image`) and, for templates only, the
public template `image`. A custom or saved image's name is never sent.
