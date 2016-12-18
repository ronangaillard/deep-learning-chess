import chess.pgn
from bitarray import bitarray

data_file = 'data/small_database.pgn'
j = 0

def game_list(filename):
    """Generator that yields games in file indicated by filename

    Args:
        filename (str): the file path of the file to which games should be extracted

    Returns:
        single game extracted from file
    """
    pgn = open(filename)

    while True:
        game = chess.pgn.read_game(pgn)

        if not game:
            pgn.close()
            break

        yield game


if __name__ == '__main__':
    with open("data.bin", 'wb') as binfile:

        for game in game_list(data_file):
            current_board = game.board()

            for move in game.main_line():
                input_vector = bitarray(12*64) # 64 cases pour 12 pieces au total

                # Ordre des pieces:
                # 1 - tour blanche R
                # 2 - cavalier blanc N
                # 3 - fou blanc B
                # 4 - dame blanche Q
                # 5 - roi blanc K
                # 6 - pion blanc P

                piece_mapper = {'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'K': 4, 'P': 5, \
                                'r': 6, 'n': 7, 'b': 8, 'q': 9, 'k': 10, 'p': 11}

                for i in range(0, 64):
                    current_piece = str(current_board)[2*i]
                    if current_piece == '.':
                        continue
                    input_vector[piece_mapper[current_piece]*64 + i] = True

                # Save current position to file
                input_vector.tofile(binfile)

                # Go to next move in game
                current_board.push(move)

            # Used to indicate end of game
            bitarray(12*64).tofile(binfile)

            j += 1

    print "Done ! Generated", j, " games !"


