from .. import ArgumentBasedParameterBlock


class SlurmParameterBlock(ArgumentBasedParameterBlock):
    title = "Slurm Parameters"

    arguments = [
        ("Use Slurm", "slurm", False),
        ("Slurm Partition", "slurm_partition", "gpu"),
    ]