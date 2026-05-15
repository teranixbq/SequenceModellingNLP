import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d_output_length(sequence_length, kernel_size):
    padding = kernel_size // 2
    return sequence_length + (2 * padding) - kernel_size + 1


class CharCNN(nn.Module):
    def __init__(
        self,
        char_vocab_size,
        char_embed_dim=64,
        num_filters=100,
        kernel_sizes=None,
        num_classes=2,
        dropout=0.5,
        pooling_type="max",
        adaptive_output_size=4,
        sequence_length=500,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5]

        self.pooling_type = pooling_type
        self.embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_embed_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        ])
        self.adaptive_pool = nn.AdaptiveMaxPool1d(adaptive_output_size)

        if pooling_type == "adaptive":
            fc_input_dim = len(kernel_sizes) * num_filters * adaptive_output_size
        elif pooling_type == "none":
            fc_input_dim = sum(
                num_filters * _conv1d_output_length(sequence_length, kernel_size)
                for kernel_size in kernel_sizes
            )
        else:
            fc_input_dim = len(kernel_sizes) * num_filters

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_results = []

        for conv in self.convs:
            conv_out = F.relu(conv(x))

            if self.pooling_type == "max":
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2])
                pooled = pooled.squeeze(2)
            elif self.pooling_type == "avg":
                pooled = F.avg_pool1d(conv_out, kernel_size=conv_out.shape[2])
                pooled = pooled.squeeze(2)
            elif self.pooling_type == "adaptive":
                pooled = self.adaptive_pool(conv_out)
                pooled = pooled.reshape(pooled.size(0), -1)
            elif self.pooling_type == "none":
                pooled = conv_out.reshape(conv_out.size(0), -1)
            else:
                raise ValueError(
                    "pooling_type harus 'max', 'avg', 'adaptive', atau 'none'"
                )

            conv_results.append(pooled)

        x = torch.cat(conv_results, dim=1)
        x = self.dropout(x)

        return self.fc(x)


class HierarchicalCharCNN(nn.Module):
    def __init__(
        self,
        char_vocab_size,
        char_embed_dim=64,
        num_filters=100,
        kernel_sizes=None,
        num_classes=2,
        dropout=0.5,
        pooling_type="max",
        adaptive_output_size=4,
        sequence_length=500,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.pooling_type = pooling_type
        self.embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList()

        in_channels = char_embed_dim
        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            in_channels = num_filters

        self.adaptive_pool = nn.AdaptiveMaxPool1d(adaptive_output_size)

        if pooling_type == "adaptive":
            fc_input_dim = num_filters * adaptive_output_size
        elif pooling_type == "none":
            output_length = sequence_length
            for kernel_size in kernel_sizes:
                output_length = _conv1d_output_length(output_length, kernel_size)
            fc_input_dim = num_filters * output_length
        else:
            fc_input_dim = num_filters

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)

        for conv in self.convs:
            x = F.relu(conv(x))

        if self.pooling_type == "max":
            x = F.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
        elif self.pooling_type == "avg":
            x = F.avg_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
        elif self.pooling_type == "adaptive":
            x = self.adaptive_pool(x)
            x = x.reshape(x.size(0), -1)
        elif self.pooling_type == "none":
            x = x.reshape(x.size(0), -1)
        else:
            raise ValueError(
                "pooling_type harus 'max', 'avg', 'adaptive', atau 'none'"
            )

        x = self.dropout(x)

        return self.fc(x)
