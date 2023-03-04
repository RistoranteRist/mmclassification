import torch.nn as nn


class CLIPTextEncoder(nn.Module):

    def __init__(self, model_name='ViT-L-14-336', pretrained='openai'):
        super().__init__()
        import open_clip
        self.tokenizer = open_clip.get_tokenizer(model_name)
        pretrained_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        # delete visual params
        pretrained_model.visual = nn.Identity()
        self.clip = pretrained_model

    @property
    def device(self):
        return self.clip.device

    @property
    def dtype(self):
        return self.clip.dtype

    def forward(self, text):
        # tokenize
        if isinstance(text, str):
            text = [text]
        text = self.tokenizer(text).to(next(self.clip.parameters()).device)
        text_features = self.clip.encode_text(text)
        return text_features
