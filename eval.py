from demo import get_args, eval_saes
import time
import os
import dictionary_learning.dictionary_learning.utils as utils
import demo_config
import click


@click.command()
@click.option("--save_dir", help="top level sae output dir")
@click.option("--model_name", default="google/gemma-4-E4B", help="base model name")
@click.option("--device", default="cuda", help="which device, cuda or cpu")
def main(save_dir, model_name, device):

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    start_time = time.time()

    ae_paths = utils.get_nested_folders(save_dir)

    eval_saes(
        model_name,
        ae_paths,
        demo_config.eval_num_inputs,
        device,
        overwrite_prev_results=True,
    )

    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
