import aegnn.asyncronous
import aegnn.utils
import aegnn.callbacks
import aegnn.datasets
import aegnn.models

try:
    import aegnn.visualize
except ModuleNotFoundError:
    import logging
    logging.warning("AEGNN Module imported without visualization tools")

# Set log level and format for the whole library.
import logging
logging.basicConfig(format='[%(asctime)s %(levelname)s]{%(filename)s:%(lineno)d} %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)

# Setup default values for environment variables, if they have not been defined already.
# Consequently, when another system is used, other than the default system, the env variable
# can simply be changed prior to importing the `aegnn` module.
aegnn.utils.io.setup_environment({
    "AEGNN_DATA_DIR": "/data/storage/simonschaefer/",
    "AEGNN_DATASET_DIR": "/data/storage/datasets",
    "AEGNN_LOG_DIR": "/data/scratch/simonschaefer/"
})
