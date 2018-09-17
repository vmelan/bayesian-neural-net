import os 
import shutil 
import logging 

def clean_events(config):
	""" Remove tensorboard events of current experiment (if any) """
	logger = logging.getLogger(__name__)
	try: 
		current_path = os.path.dirname(os.path.abspath(__file__))
		logger.info("Removing existing events of current experiment")
		# deleting the folder with its content then creating an empty one
		shutil.rmtree(os.path.join(current_path, 'tensorboard/' + config['exp_name'] + "/"))
		os.mkdir(os.path.join(current_path, 'tensorboard/' + config['exp_name'] + "/"))
	except:
		pass

