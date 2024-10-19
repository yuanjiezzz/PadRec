import argparse

from recbole.quick_start import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    # added by me
    parser.add_argument(
        "--num_interactions_to_add", 
        type=int, 
        default=0, 
        help="Number of interactions to add (DiffRec) ",
    )

    parser.add_argument(
        "--noise_scale", 
        type=float, 
        default=0.001, 
        help="noise scale for noise generating (DiffRec) ",
    )

    parser.add_argument(
        "--noise_min", 
        type=float, 
        default=0.0005, 
        help="noise lower bound for noise generating (DiffRec) ",
    )

    parser.add_argument(
        "--noise_max", 
        type=float, 
        default=0.005, 
        help="noise upper bound for noise generating (DiffRec) ",
    )


    def str_to_bool(value):
        return value.lower() == 'true'

    parser.add_argument(
        "--update_noise",
        type=str_to_bool,
        default=False,
        help="Set to 'True' to enable noise update based on freq (DiffRec)."
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
        num_interactions_to_add=args.num_interactions_to_add,
        noise_scale=args.noise_scale,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        update_noise=args.update_noise
    )
