import torch
from transformers import ASTFeatureExtractor, ASTModel, ASTConfig

def load_AST():
  config = ASTConfig(max_length=128)
  AST_SAMPLE_RATE = 16000
  feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=config)
  feature_extractor_orig = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
  model_orig = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
  model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=config,ignore_mismatched_sizes=True)
  model.embeddings.position_embeddings = torch.nn.Parameter(model_orig.embeddings.position_embeddings[0,:146,:])

  return model