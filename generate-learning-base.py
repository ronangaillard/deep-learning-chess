import chess.pgn

data_file = 'data/2016.pgn'


i = 0

def parse_game(game):
    while not node.is_end():
        next_node = node.variation(0)
        node = next_node


def game_list(filename):
    pgn = open(filename)

    while True:
        game = chess.pgn.read_game(pgn)

        if not game:
            pgn.close()
            break

        yield game


for game in game_list(data_file):
    print i
    i += 1


print i, "games read"

