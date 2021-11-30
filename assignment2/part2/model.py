###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################

import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """

    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Parameters
        ----------
        lstm_hidden_dim : int
            hidden state dimension.
        embedding_size : int
            size of embedding (and hence input sequence).
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # embedding weights
        self.w_gx = nn.Parameter(
            torch.zeros(self.hidden_dim, self.embed_dim), requires_grad=True
        )
        self.w_ix = nn.Parameter(
            torch.zeros(self.hidden_dim, self.embed_dim), requires_grad=True
        )
        self.w_fx = nn.Parameter(
            torch.zeros(self.hidden_dim, self.embed_dim), requires_grad=True
        )
        self.w_ox = nn.Parameter(
            torch.zeros(self.hidden_dim, self.embed_dim), requires_grad=True
        )
        # hidden weights
        self.w_gh = nn.Parameter(
            torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True
        )
        self.w_ih = nn.Parameter(
            torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True
        )
        self.w_fh = nn.Parameter(
            torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True
        )
        self.w_oh = nn.Parameter(
            torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True
        )
        # bias
        self.b_g = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Parameters
        ----------
        self.parameters: list of all parameters.
        self.hidden_dim: hidden state dimension.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for param in self.parameters():
            # initialize weights as specified by inst
            nn.init.uniform_(
                param, -1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)
            )
        # add one to forget gate bias
        self.b_f += 1
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Parameters
        ----------
        embeds : array-like
            embedded input sequence with shape
            (input length, batch size, embedding_size)

        Returns
        -------
        output : array-like
            output of LSTM with shape
            (input length, batch size, hidden_size)
        """
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # initialize hidden state
        output = torch.zeros_like(embeds)
        h = torch.zeros(embeds.shape[1], self.hidden_dim)
        c = torch.zeros(embeds.shape[1], self.hidden_dim)
        for i, seq_el in enumerate(embeds):
            g = torch.tanh(seq_el @ self.w_gx.T + h @ self.w_gh.T + self.b_g)
            i = torch.sigmoid(seq_el @ self.w_ix.T + h @ self.w_ih.T + self.b_i)
            f = torch.sigmoid(seq_el @ self.w_fx.T + h @ self.w_fh.T + self.b_f)
            o = torch.sigmoid(seq_el @ self.w_ox.T + h @ self.w_oh.T + self.b_o)
            c = g * i + c * f
            h = torch.tanh(c) * o
            output[i] = h
        return output
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """

    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.0):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################
