from .one_fits_all.gpt4ts import gpt4ts
from .linear.linear_model import Linear_Model
from .mlp.mlp_model import MLP_Model
from .rnn.rnn_model import RNN_Model
from .SingleCNN.cnn import SinglePulseCNN
from .gru.gru_model import PulseGRUModel
from .mlp_cnn.mlp_cnn import PulseConvModel
from .transformer.transformer_model import PulseTransformerModel

model_factory = {
    "one_fits_all": gpt4ts,
    "linear": Linear_Model,
    "mlp": MLP_Model,
    "rnn": RNN_Model,
    "1dcnn": SinglePulseCNN,
    "gru": PulseGRUModel,
    "mlp_cnn": PulseConvModel,
    "transformer": PulseTransformerModel,
    # Add other models here as needed
}
