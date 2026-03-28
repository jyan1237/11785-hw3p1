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

        # Implement the greedy search decoding algorithm here
        _, seq_len, batch_size = y_probs.shape

        y_best_probs_ind = np.argmax(y_probs, axis=0)
        path_prob = np.prod(np.max(y_probs, axis=0), axis=0)

        for b in range(batch_size):
            path = y_best_probs_ind[:, b]

            # remove repeats
            mask = path[1:] != path[:-1]    # mask from indices 1 to end where its different from prev element
            mask = np.insert(mask, 0, True) # first element is by default different
            path = path[mask]

            # remove blanks
            path = path[path != 0]

            # convert indices to string
            string = "".join([self.symbol_set[i-1] for i in path])
            decoded_path.append(string)
            
        return decoded_path[0], path_prob[0]


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
        merged_paths, best_path, best_path_score = {}, "", 0.0

        # TODO: Implement the beam search decoding algorithm here. This typically involves:
        # 1. Initializing a set of paths with their probabilities.
        # 2. For each time step, extending existing paths with all possible symbols (handling the three cases for repeats/blanks/new symbols)
        # 3. Merging paths that produce the same decoded sequence by summing probabilities
        # 4. Pruning the set of paths to keep only the top 'beam_width' paths
        # 5. After iterating all time steps, merge duplicate paths again if needed
        # 6. Return the best final sequence and the paths & scores for all final sequences
        full_symbol_set = ['-'] + self.symbol_set
        active_paths = {"-": 1.0}
        temp_paths = {}

        for t in range(T):
            now_prob = y_probs[:, t, 0]
            active_paths = dict(sorted(active_paths.items(), key=lambda item: item[1], reverse=True)[:self.beam_width])
            
            for path, score in active_paths.items():
                for i, sym in enumerate(full_symbol_set):
                    # compute new path based on last symbol of path
                    last_sym = path[-1]
                    if last_sym == sym:
                        new_path = path
                    elif last_sym == '-':
                        new_path = path[:-1] + sym
                    else:
                        new_path = path + sym
                    
                    # computing new score
                    new_score = score * now_prob[i]

                    if new_path in temp_paths:
                        temp_paths[new_path] += new_score
                    else:
                        temp_paths[new_path] = new_score
            
            active_paths = temp_paths
            temp_paths = {}

        # final merging of paths
        for path, score in active_paths.items():
            # remove any blanks that might be remaining
            clean_path = path.strip('-')
            
            if clean_path in merged_paths:
                merged_paths[clean_path] += score
            else:
                merged_paths[clean_path] = score

        # Find the best path after all merging is complete
        for path, score in merged_paths.items():
            if score > best_path_score:
                best_path = path
                best_path_score = score
        
        return best_path, merged_paths
