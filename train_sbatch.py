from simple_slurm import Slurm

### LRZ ###

mlflow_info_slots = {
    # '7': 'uv run /dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/train.py --mlflow_info_slot 7 --model dit --scheduler flow_matching --mask_conditioning 64
    # '8': 'uv run /dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/train.py --mlflow_info_slot 8 --model unet --scheduler flow_matching --mask_conditioning 64',
}

for mlflow_info_slot, command_ in mlflow_info_slots.items():

    slurm = Slurm(
        job_name=f'vb_ldm_{mlflow_info_slot}',
        qos='mcml',
        output='/dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/slurm/%x-%A.out',
        error='/dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/slurm/%x-%A.err',
        partition='mcml-hgx-a100-80x4',
        time='2-0:00:00',
        nodes=1,
        ntasks_per_node=1,
        gres='gpu:1',
        cpus_per_task=16,
        mem='128G',
        dependency='singleton'
    )

    command = f"""
    cd /dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm
    {command_}
    """

    for i in range(3):
        job_id = slurm.sbatch(command)