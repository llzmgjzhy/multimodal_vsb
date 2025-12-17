from .one_fits_all.gpt4ts import gpt4ts
from .linear.linear_model import Linear_Model

model_factory = {
    "one_fits_all": gpt4ts,
    "linear":Linear_Model,
    # Add other models here as needed
}
