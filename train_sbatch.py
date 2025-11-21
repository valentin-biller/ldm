from simple_slurm import Slurm

### LRZ ###

mlflow_info_slots = {
    # '6': 'uv run /dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/train.py --mlflow_info_slot 6 --model unet --scheduler ddpm --mask_conditioning 32 --latent_shape 4,32,32,20 --save_samples_every 0',
    # '7': 'uv run /dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/train.py --mlflow_info_slot 7 --model dit --scheduler ddpm --mask_conditioning 32 --latent_shape 4,32,32,20 --save_samples_every 0',
    # '8': 'uv run /dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/train.py --mlflow_info_slot 8 --model unet --scheduler iddpm --mask_conditioning 32 --latent_shape 4,32,32,20 --save_samples_every 0',
    '9': 'uv run /dss/dsshome1/01/ge65mod2/valentin/master-thesis/ldm/train.py --mlflow_info_slot 9 --model dit --scheduler iddpm --mask_conditioning 32 --latent_shape 4,32,32,20 --save_samples_every 0',
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