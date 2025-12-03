import torch
import torch.nn.functional as F

class KeypointComparator:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.model.eval()

    @torch.no_grad()
    def compare(self, query_img, candidate_imgs):
        query_feat = self.model(query_img).flatten()
        similarities = []
        for img in candidate_imgs:
            feat = self.model(img.unsqueeze(0)).flatten()
            sim = F.cosine_similarity(query_feat, feat, dim=0)
            similarities.append(sim.item())
        best_idx = int(torch.tensor(similarities).argmax())
        return best_idx
