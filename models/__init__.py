from .one_fits_all.gpt4ts import gpt4ts
from .linear.linear_model import Linear_Model
from .mlp.mlp_model import MLP_Model
from .rnn.rnn_model import RNN_Model

model_factory = {
    "one_fits_all": gpt4ts,
    "linear": Linear_Model,
    "mlp": MLP_Model,
    "rnn": RNN_Model,
    # Add other models here as needed
}
