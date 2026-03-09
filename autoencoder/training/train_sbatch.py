from simple_slurm import Slurm

slurm = Slurm(
    job_name='vb_ae',
    qos='mcml',
    output='/slurm/%x-%A.out',
    error='/slurm/%x-%A.err',
    partition='mcml-hgx-h100-94x4',
    time='2-0:00:00',
    nodes=1,
    ntasks_per_node=1,
    gres='gpu:1',
    cpus_per_task=16,
    mem='128G',
    dependency='singleton'
)

command = """
cd ...
uv run ...
"""

for i in range(3):
    job_id = slurm.sbatch(command)