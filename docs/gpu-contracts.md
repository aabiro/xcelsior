# Whole-GPU rentals

The scheduler allocates physical GPU UUIDs. A renter receives each allocated
GPU for the duration of the job. Providers choose how many whole GPU slots to
offer for spot. Slots kept out of spot remain eligible for other placement
according to the host's availability settings. Memory capacity and minimum
memory requirements are hardware and placement facts. The listing and launch
APIs return HTTP 400 for fractional GPU counts and VRAM-budget contracts,
including on the RTX 2060, RTX 3060, and GeForce RTX 5090.

The agent reports outside compute use about once a minute by default. It
excludes graphics clients and compute processes it recognizes as its own.
Placement and the reservation transaction skip a device with a positive
outside process count. VRAM telemetry on its own leaves it eligible. A
compute process may start during the interval between observations, so the
worker checks the leased devices again on start. The exclusive GPU and
Dedicated tiers also acquire hardware scope locks and check occupancy.

If NVIDIA's process listing is unavailable, the updated fenced worker refuses
the start. Operators can list trusted internal hosts in
`XCELSIOR_GPU_OCCUPANCY_TRUSTED_HOSTS` to let them start when the listing fails.
A positive compute process count still refuses the start. The API counts
observations with unavailable process listings in
`xcelsior_worker_gpu_outside_use_unavailable_total`. An earlier worker has
to be upgraded before it can enforce the start policy.

A clean-to-busy report on an already leased GPU is an SLA and reputation
event. The running container continues. The next placement excludes the
device while the reported outside compute use persists. The checks and locks
depend on a cooperating host and cannot block arbitrary host administrator
processes from opening the GPU.

The Headmesh settings `gpu = "shared"` and `nvidia.com/gpu=all` expose devices
to a container. They impose no VRAM fence.

**Theory (trusted renter only).** On a GeForce RTX 5090,
`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` or a cooperative allocation wrapper could
limit Xcelsior's own workload under its own process controls. An untrusted
renter can bypass a cooperative limit. Third-party listings use whole GPUs.
[NVIDIA MPS environment variables](https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html)
