import os
import torch


def pick_device(config, logger):
	""" Pick device """ 
	if config["cuda"] and not torch.cuda.is_available():
		logger.warning("Warning: There's no CUDA support on this machine,"
			"training is performed on cpu.")
		device = torch.device("cpu")
	elif not config["cuda"] and torch.cuda.is_available():
		logger.info("Training is performed on cpu by user's choice")
		device = torch.device("cpu")
	elif not config["cuda"] and not torch.cuda.is_available():
		logger.info("Training on cpu")
		device = torch.device("cpu")
	else:
		logger.info("Training on gpu")
		device = torch.device("cuda:" + str(config["gpu"]))	

	return device 

def ensure_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)