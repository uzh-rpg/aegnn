import aegnn.callbacks
import aegnn.datasets
import aegnn.filters
import aegnn.models
import aegnn.transforms
import aegnn.utils

try:
    import aegnn.visualize
except ModuleNotFoundError:
    print("AEGNN Module imported without visualization tools")


# Setup default values for environment variables, if they have not been defined already.
# Consequently, when another system is used, other than the default system, the env variable
# can simply be changed prior to importing the `aegnn` module.
aegnn.utils.setup_environment({
    "AEGNN_DATA_DIR": "/data/storage/simonschaefer/",
    "AEGNN_LOG_DIR": "/data/scratch/simonschaefer/"
})
