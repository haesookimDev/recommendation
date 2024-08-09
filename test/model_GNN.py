import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TravelRecommendationGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(TravelRecommendationGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv6(x, edge_index)

        x = F.normalize(x, p=2, dim=1)

        return x
    

class TravelRecommendationModel(torch.nn.Module):
    def __init__(self, gnn, hidden_channels, num_classes, num_heads=4):
        super(TravelRecommendationModel, self).__init__()
        self.gnn = gnn
        self.attention = torch.nn.MultiheadAttention(hidden_channels, num_heads)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, traveler_idx, trip_idx, dest_idx):
        embeddings = self.gnn(x, edge_index)
        
        traveler_emb = embeddings[traveler_idx].unsqueeze(1) # [batch_size, 1, hidden_channels]
        trip_emb = embeddings[trip_idx].unsqueeze(1) # [batch_size, 1, hidden_channels]
        dest_emb = embeddings[dest_idx] # [num_destinations, hidden_channels]
        
        # 어텐션 메커니즘을 사용하여 traveler와 trip 임베딩 결합
        combined_emb, _ = self.attention(traveler_emb, trip_emb, trip_emb)
        combined_emb = combined_emb.squeeze(1) # [batch_size, hidden_channels]

         # combined_emb를 dest_emb와 같은 shape로 확장
        combined_emb = combined_emb.unsqueeze(1)  # [batch_size, 1, hidden_channels]
        dest_emb = dest_emb.unsqueeze(0)  # [1, num_destinations, hidden_channels]

        # 코사인 유사도 계산
        similarity = F.cosine_similarity(combined_emb, dest_emb, dim=2)
        
        # 점수 예측
        scores = self.fc(dest_emb.squeeze(0))
        
        return similarity, scores, embeddings
    
class ContrastiveLoss():
    def __init__(self, similarity, labels, temperature=0.5):
        self.similarity = similarity
        self.labels = labels
        self.temperature = temperature
    def loss(self):
        exp_sim = torch.exp(self.similarity / self.temperature)
        pos_sum = torch.sum(exp_sim * self.labels, dim=1)
        neg_sum = torch.sum(exp_sim * (1 - self.labels), dim=1)
        loss = -torch.log(pos_sum / (pos_sum + neg_sum)).mean()
        return loss