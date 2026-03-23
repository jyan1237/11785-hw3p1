import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO: Implement the greedy search decoding algorithm here

        #return decoded_path, path_prob
        raise NotImplementedError


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        best_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_paths [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        merged_paths, best_path, best_path_score = None, None, None

        # TODO: Implement the beam search decoding algorithm here. This typically involves:
        # 1. Initializing a set of paths with their probabilities.
        # 2. For each time step, extending existing paths with all possible symbols (handling the three cases for repeats/blanks/new symbols)
        # 3. Merging paths that produce the same decoded sequence by summing probabilities
        # 4. Pruning the set of paths to keep only the top 'beam_width' paths
        # 5. After iterating all time steps, merge duplicate paths again if needed
        # 6. Return the best final sequence and the paths & scores for all final sequences
        
        #return best_path, merged_paths
        raise NotImplementedError