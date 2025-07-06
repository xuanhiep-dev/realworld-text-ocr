import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import time
import matplotlib.pyplot as plt


class CRNNTrainer(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers,
        dropout=0.2, unfreeze_layers=3, blank_char='-',
        device='cuda'
    ):
        super(CRNNTrainer, self).__init__()
        self.device = device
        self.blank_char = blank_char

        # ---- Backbone ----
        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze last layers
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            hidden_size, hidden_size,
            n_layers, bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),
            nn.LogSoftmax(dim=2)
        )

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # CTC expects (T, B, C)
        return x

    def evaluate(self, dataloader, criterion):
        self.eval()
        losses = []
        with torch.no_grad():
            for inputs, labels, labels_len in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels_len = labels_len.to(self.device)

                outputs = self.forward(inputs)
                logits_lens = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(self.device)

                loss = criterion(outputs, labels, logits_lens, labels_len)
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def fit(
        self, train_loader, val_loader,
        criterion, optimizer, scheduler,
        epochs=10, max_grad_norm=2
    ):
        self.to(self.device)
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            start = time.time()
            batch_train_losses = []

            self.train()
            for inputs, labels, labels_len in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels_len = labels_len.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                logits_lens = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(self.device)

                loss = criterion(
                    outputs, labels.cpu(), logits_lens.cpu(), labels_len.cpu()
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_grad_norm)
                optimizer.step()

                batch_train_losses.append(loss.item())

            train_loss = sum(batch_train_losses) / len(batch_train_losses)
            val_loss = self.evaluate(val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\t"
                f"Val loss: {val_loss:.4f}\tTime: {time.time() - start:.2f}s"
            )

            scheduler.step()

        self.train_losses = train_losses
        self.val_losses = val_losses
        return train_losses, val_losses

    def show_stats(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(self.train_losses)
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        ax[1].plot(self.val_losses, color="orange")
        ax[1].set_title("Validation Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")

        plt.show()

    def predict(self, img):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(img.to(self.device))
        return outputs

    def decode_label(self, encoded_sequences, idx_to_char):
        decoded_sequences = []
        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None
            for token in seq:
                if token != 0:
                    char = idx_to_char[token.item()]
                    if char != self.blank_char:
                        if char != prev_char or prev_char == self.blank_char:
                            decoded_label.append(char)
                    prev_char = char
            decoded_sequences.append("".join(decoded_label))
        return decoded_sequences
