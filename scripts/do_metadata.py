from msst.utils.dataset import prepare_data
from msst.utils.settings import get_model_from_config, parse_args_train

args = parse_args_train(None)
_, config = get_model_from_config(args.model_type, args.config_path)

batch_size = config.training.batch_size * args.device_ids

prepare_data(config, args, batch_size)