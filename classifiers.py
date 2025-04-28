from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- CNN-LSTM Model (Simple) for Feature Inputs ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # (batch, seq, feature)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def train_classifier(X_train, y_train, X_test, y_test, classifier_name="cnn-lstm"):
    """Train different classifiers based on input."""
    
    if classifier_name.lower() == "svm":
        model = SVC()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    elif classifier_name.lower() == "rf":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    elif classifier_name.lower() == "xgb":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    elif classifier_name.lower() == "cnn-lstm":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_LSTM_Model(input_dim=X_train.shape[1]).to(device)
        
        # Prepare Data
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(10):  # simple training for demo
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor)
            preds = preds.argmax(dim=1).cpu().numpy()
    
    else:
        raise ValueError("Unsupported classifier!")
    
    from utils.evaluation_metrics import calculate_metrics
    performance = calculate_metrics(y_test, preds)
    
    return model, performance
