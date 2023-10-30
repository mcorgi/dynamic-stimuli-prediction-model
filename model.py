from bitsandbytes.nn import Linear4bit
from torch.nn import Linear
import torch
import torch.nn as nn
from transformers import VivitImageProcessor, VivitModel, VivitForVideoClassification
from transformers.modeling_outputs import ImageClassifierOutput
        

class Swappable(nn.Module):
    
    def __init__(self, named_layers: dict[str, nn.Module]):
        super().__init__()
        self.named_layers = named_layers
        self._named_layers = nn.ModuleList(named_layers.values())
    
    def forward(self, x, name):
        return self.named_layers[name](x)

def pad_to_nearest_multiple(tensor, target_multiple):
    """Pads a tensor to the nearest multiple along the second dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor to pad.
        target_multiple (int): The target multiple to pad the second dimension to.
        
    Returns:
        torch.Tensor: Padded tensor.
    """
    batch_size, seq_len, hidden_size = tensor.size()
    
    # Calculate the padding size needed to reach the nearest multiple
    padding_size = (target_multiple - (seq_len % target_multiple)) % target_multiple
    
    # Calculate the padding for the left and right sides
    left_padding = padding_size // 2
    right_padding = padding_size - left_padding
    
    # Create padding tuple
    padding_tuple = (0, 0, left_padding, right_padding)
    
    # Apply padding
    padded_tensor = torch.nn.functional.pad(tensor, padding_tuple, 'constant', 0)
    
    return padded_tensor

class Reducer(nn.Module):
    
    def __init__(self, input_seq_len=3137, target_seq_len=32, hidden_dim=768):
        super().__init__()
        self.target_seq_len = target_seq_len
        padded_len = input_seq_len if input_seq_len % target_seq_len  == 0 else (input_seq_len // target_seq_len + 1) * 32
        stride = padded_len // target_seq_len
        kernel_size = stride
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=stride)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        '''
        x has size batch_size x seq_len x hidden_dim
        '''
        x = pad_to_nearest_multiple(x, self.target_seq_len)
        print('x', x)
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x

class ReducerClassifier(nn.Module):
    
    def __init__(self, out_dim, input_seq_len=3137, target_seq_len=32, hidden_dim=768):
        super().__init__()
        self.reducer = Reducer(input_seq_len=input_seq_len, target_seq_len=target_seq_len, hidden_dim=hidden_dim)
        self.linear = Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.reducer(x)
        x = self.linear(x)
        return x
    
    
class SensoriumVivitForVideoClassification(VivitForVideoClassification):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.reducer = Re
        self.reducer = Reducer()
    
    def forward(
        self,
        pixel_values=None,
        mouse=None,
        inputs_embeds=None,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vivit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

#         sequence_output = outputs[0]

#         logits = self.classifier(sequence_output[:, 0, :])
        print('last hidden', outputs.last_hidden_state)

        reduced = self.reducer(outputs.last_hidden_state)
        print('reduced', reduced)

        logits = self.classifier(reduced, mouse)
        print('logits', logits)
        

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )