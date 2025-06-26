from transformers import CLIPModel,BertConfig
from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MV_CLIP(nn.Module):
    def __init__(self, args):
        super(MV_CLIP, self).__init__()
        # Loading pretrained CLIP
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Loads a default configuration of a BERT model (usually used for NLP task)
        self.config = BertConfig.from_pretrained("bert-base-uncased")

        # Modifies BERT configuration
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8

        # Initializes a MultiModalEncoder (transformer based module) using modified BERT config
        # and specifies the number of transformer layers, taken from args
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)

        # When simple linear transformation is done to text and image.
        if args.simple_linear:
            # nn.Linear(in_features, out_features). Applies simple linear transformations to text and image
            self.text_linear =  nn.Linear(args.text_size, args.text_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        # When more complex transformation is applied to text and image.
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.text_size), # Linear transformation layer
                nn.Dropout(args.dropout_rate), # Dropout layer to prevent overfitting
                nn.GELU() # GELU Activation function layer
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        # Classifier for fused multimodal reps., text reps., and image reps.
        # The in_features and out_features here are structured based on how feature representations from the CLIP model
        # are processed before classification.
        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        # Attention mechanism: A linear layer that maps from text_size to a single attention score
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, labels):
        # Runs the CLIP model on the given inputs (which contain text and image). 
        # The model returns a dictionary that contains text_encoder_output,and image_encoder_output.
        # It also ensures that attention values are also returned.
        output = self.model(**inputs,output_attentions=True)

        # Extracts hidden states (token embeddings and patch embeddings) from the text encoder and image encoder  
        text_features = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']

        # Extracts the pooled representations. (single vector summarizing the entire text and another single vector representing the entire image)
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']

        # Passes the pooled features through the linear transformation layers
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        # Projects text token embeddings and image token embeddings into CLIP embedding space.
        text_embeds = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)

        # Concatenates the image and text embeddings along the sequence dimension.
        # This forms a combined multimodel representation.
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)

        # Creates a combined attention mask.
        attention_mask = torch.cat((torch.ones(text_features.shape[0], 50).to(text_features.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Passing throught multimodal transformer
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask, output_all_encoded_layers=False)
        
        # Selec the last layer hidden state as the final representation
        fuse_hiddens = fuse_hiddens[-1]

        # Extracting new text and image representation
        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        # Computes attention weight from new text and image feature.
        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)

        # Stack text and image weight together    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        # Split attention weights into text weight and image weight
        tw, iw = att.split([1,1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
   
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)

        score = fuse_score + text_score + image_score

        outputs = (score,)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            loss_text = self.loss_fct(logits_text, labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss = loss_fuse + loss_text + loss_image

            outputs = (loss,) + outputs
        return outputs


