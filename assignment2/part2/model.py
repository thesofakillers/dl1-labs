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
        self.w_x = nn.Parameter(
            torch.zeros(4 * self.hidden_dim, self.embed_dim), requires_grad=True
        )
        self.w_h = nn.Parameter(
            torch.zeros(4 * self.hidden_dim, self.hidden_dim), requires_grad=True
        )
        # bias
        self.b_g = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        # cell and hidden state
        self.h = None
        self.c = None
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
        with torch.no_grad():
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

        # initialize output, ensuring its on the same device as embeds
        output = torch.zeros(
            (embeds.shape[0], embeds.shape[1], self.hidden_dim), device=self.w_x.device
        )
        # initialize hidden and cell states if necessary
        if self.h is None:
            self.h = torch.zeros(
                embeds.shape[1], self.hidden_dim, device=self.w_x.device
            )
        if self.c is None:
            self.c = torch.zeros(
                embeds.shape[1], self.hidden_dim, device=self.w_h.device
            )
        # LSTM computation
        for j, seq_el in enumerate(embeds):
            biases = torch.hstack((self.b_i, self.b_f, self.b_o, self.b_g))
            opt_mult = seq_el @ self.w_x.T + self.h @ self.w_h.T + biases
            i, f, o, g = torch.chunk(opt_mult, 4, dim=1)
            g = torch.tanh(g)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            self.c = g * i + self.c * f
            self.h = torch.tanh(self.c) * o
            output[j] = self.h
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

        Parameters
        ----------
        args.vocabulary_size: int
            The size of the vocabulary.
        args.embedding_size: int
            The size of the embedding.
        args.lstm_hidden_dim: int
            The dimension of the hidden state in the LSTM cell.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.embedding = nn.Embedding(args.vocabulary_size, args.embedding_size)
        self.lstm = LSTM(args.lstm_hidden_dim, args.embedding_size)
        self.linear = nn.Linear(args.lstm_hidden_dim, args.vocabulary_size)
        self.vocabulary_size = args.vocabulary_size
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : array-like
            input sequence with shape (seq_len, batch_size)
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # embedding gives (seq_len, batch_size, embedding_size)
        x = self.embedding(x)
        # lstm forward gives (seq_len, batch_size, hidden_size)
        x = self.lstm(x)
        # linear forward gives (seq_len, batch_size, vocabulary_size)
        out = self.linear(x)
        return out
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.0):
        """
        Sampling from the text generation model.

        Parameters
        ----------
        batch_size : int
            Number of samples to return
        sample_length : int
            length of desired sample.
        temperature: float
            temperature of the sampling process (see exercise sheet for definition).

        Returns
        -------
        samples : array-like
            samples with shape (sample_length, batch_size)
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # initialize characters randomly
        chars = torch.randint(self.vocabulary_size, (sample_length, batch_size))

        # overwrite all chars except first with sampled characters
        for step in range(1, sample_length):
            # use previous step, hidden/cell state are stored and kept track of BTS
            pred = self.forward(chars[step - 1])[-1]
            # pred is of shape (1, batch_size, vocabulary_size)
            if temperature == 0:
                # argmax gives (1, batch_size)
                new_char = pred.argmax(dim=-1)
            else:
                # randomly sample using temperature-scaled softmax weights
                new_char = torch.multinomial(
                    torch.softmax(pred / temperature, dim=-1), 1
                )
            # save new char to the current step
            chars[step] = new_char

        return chars
        #######################
        # END OF YOUR CODE    #
        #######################


if __name__ == "__main__":
    hidden_dim = 5
    embed_size = 10
    batch_n = 3

    lstm = LSTM(hidden_dim, embed_size)
    out = lstm.forward(torch.randn(2, batch_n, embed_size))
    print(out.shape)
