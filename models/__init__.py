from .one_fits_all.gpt4ts import gpt4ts
from .linear.linear_model import Linear_Model
from .mlp.mlp_model import PulseMLPBaseline, MLP_Model
from .rnn.rnn_model import RNN_Model
from .SingleCNN.cnn import SinglePulseCNN
from .gru.gru_model import PulseGRUModel
from .mlp_cnn.mlp_cnn import PulseConvModel
from .transformer.transformer_model import PulseTransformerModel
from .cnn_mlp.cnn_mlp import PulseEncoder
from .prototype.prototype import ProtoModelB3,ProtoEnrichMLP
from .unsupervised_learning.prototype.prototype_stats import DL_SOTA_PrototypeNet
from .patchtst.patchtst import Model as PatchTST_Model
from .unsupervised_learning.transformer.transformer import SetEncoderTransformer, DownStreamClassifier
from .unsupervised_learning.swav.swav import SwAVModel,ClassifySwAVModel

model_factory = {
    "one_fits_all": gpt4ts,
    "linear": Linear_Model,
    "mlp": PulseMLPBaseline,
    "rnn": RNN_Model,
    "1dcnn": SinglePulseCNN,
    "gru": PulseGRUModel,
    "mlp_cnn": PulseConvModel,
    "transformer": PulseTransformerModel,
    "cnn_mlp": PulseEncoder,
    "prototype_simple": ProtoEnrichMLP,
    "prototype_stats": DL_SOTA_PrototypeNet,
    "patchtst": PatchTST_Model,
    "set_transformer": SetEncoderTransformer,
    "classifier_set_transformer": DownStreamClassifier,
    "swav": SwAVModel,
    "classifier_swav": ClassifySwAVModel,
    # Add other models here as needed
}
