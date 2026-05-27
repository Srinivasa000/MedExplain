"""
BiLSTM-CRF Model for Drug Named Entity Recognition
Combines BiLSTM for feature extraction and CRF for sequence tagging
"""

import torch
import torch.nn as nn
from config import Config

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF model for sequence tagging
    """

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # Embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Linear layer to project to tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF transition parameters
        # transitions[i][j] = transition score from tag j to tag i
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        # Initialize transitions
        self.transitions.data[tag_to_ix['O'], :] = -10000  # No transition to O
        self.transitions.data[:, tag_to_ix['O']] = -10000  # No transition from O

    def _get_lstm_features(self, sentence):
        """Get BiLSTM emission scores"""
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats, mask):
        """Forward algorithm for CRF"""
        batch_size, seq_len, tagset_size = feats.size()

        # Initialize forward variables
        init_alphas = torch.full((batch_size, tagset_size), -10000.)
        init_alphas[:, self.tag_to_ix['O']] = 0.

        forward_var = init_alphas.to(feats.device)

        # Iterate through the sequence
        for feat_idx in range(seq_len):
            alphas_t = []
            for next_tag in range(tagset_size):
                emit_score = feats[:, feat_idx, next_tag].view(batch_size, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(batch_size, 1))

            forward_var = torch.cat(alphas_t, dim=1) * mask[:, feat_idx].unsqueeze(1)
            forward_var = forward_var + forward_var * (1 - mask[:, feat_idx].unsqueeze(1))

        terminal_var = forward_var
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _score_sentence(self, feats, tags, mask):
        """Calculate score of a given tag sequence"""
        batch_size, seq_len = tags.size()
        score = torch.zeros(batch_size).to(feats.device)

        tags = torch.cat([torch.full((batch_size, 1), self.tag_to_ix['O'], dtype=torch.long).to(tags.device), tags], dim=1)

        for i in range(seq_len):
            emit_score = feats[:, i, tags[:, i + 1]]
            trans_score = self.transitions[tags[:, i + 1], tags[:, i]]
            score = score + (emit_score + trans_score) * mask[:, i]

        return score

    def _viterbi_decode(self, feats, mask):
        """Viterbi algorithm for finding best path"""
        batch_size, seq_len, tagset_size = feats.size()

        # Initialize
        init_vvars = torch.full((batch_size, tagset_size), -10000.)
        init_vvars[:, self.tag_to_ix['O']] = 0

        forward_var = init_vvars.to(feats.device)
        backpointers = []

        # Forward pass
        for feat_idx in range(seq_len):
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var, dim=1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[range(batch_size), best_tag_id])

            forward_var = (torch.stack(viterbivars_t, dim=1) + feats[:, feat_idx]) * mask[:, feat_idx].unsqueeze(1)
            forward_var = forward_var + forward_var * (1 - mask[:, feat_idx].unsqueeze(1))
            backpointers.append(bptrs_t)

        # Backtrack
        path_score, best_tag_id = torch.max(forward_var, dim=1)
        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = torch.stack(bptrs_t, dim=1)[range(batch_size), best_tag_id]
            best_path.append(best_tag_id)

        best_path.reverse()
        return path_score, torch.stack(best_path, dim=1)

    def neg_log_likelihood(self, sentence, tags, mask):
        """Calculate negative log likelihood loss"""
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return (forward_score - gold_score).mean()

    def forward(self, sentence, mask):
        """Forward pass - get best path"""
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats, mask)
        return score, tag_seq
