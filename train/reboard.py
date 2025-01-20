import re
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout, QScrollArea, QPushButton, QLineEdit, QHBoxLayout
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt


def parse_log_file(log_file):
    """Parse the log file to extract board actions, best Q-values, and MCTS actions for each episode."""
    episodes = {}
    current_episode = None
    print("Loading the log.")
    with open(log_file, 'r') as file:
        for line in file:
            if 'Episode' in line and 'INFO' in line:
                # Extract episode number and metadata
                current_episode = int(re.search(r"Episode (\d+)/", line).group(1))
                match = re.search(r"Winner=(\d+),Turn=(\d+)", line)
                winner = int(match.group(1)) if match else None
                turns = int(match.group(2)) if match else None
                episodes[current_episode] = {"actions": [], "best_q_values": [], "mcts_actions": [], "winner": winner, "turns": turns}
            elif 'Random Action SELECT' in line and current_episode is not None:
                # Extract the column from the log
                action = int(re.search(r"SELECT=(\d+)", line).group(1))
                episodes[current_episode]["actions"].append(action)
            elif 'best Q SELECT' in line and current_episode is not None:
                # Extract the column with the best Q-value
                best_q = int(re.search(r"SELECT=(\d+)", line).group(1))
                episodes[current_episode]["best_q_values"].append(best_q)
            elif 'MCTS Action SELECT' in line and current_episode is not None:
                # Extract the column with the MCTS action
                mcts_action = int(re.search(r"SELECT=(\d+)", line).group(1))
                episodes[current_episode]["mcts_actions"].append(mcts_action)
    print(f"Loaded {len(episodes)} episodes")
    return episodes

def build_board(actions, best_q_values, mcts_actions):
    """Build the board state from the actions and highlight best Q-values and MCTS actions."""
    
    rows, cols = 6, 7  # Standard Connect 4 board size
    board = np.full((rows, cols), '.', dtype=str)

    current_player = 'X'  # Player 1 starts
    last_action = None

    for i, action in enumerate(actions):
        # Place token in the first available row of the column
        for row in range(rows - 1, -1, -1):
            if board[row, action] == '.':
                if action in mcts_actions:
                    board[row, action] = '#' if current_player == 'X' else '$'  # Highlight MCTS actions
                elif action in best_q_values:
                    board[row, action] = '@'  # Highlight best Q-value
                else:
                    board[row, action] = current_player
                last_action = (row, action, current_player)  # Track the last move
                break

        # Toggle players
        current_player = 'O' if current_player == 'X' else 'X'

    return board, last_action

class Connect4Viewer(QWidget):
    def __init__(self, episodes):
        super().__init__()
        self.episodes = episodes
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Connect 4 Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.input_layout = QHBoxLayout()
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("Start Episode")
        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText("End Episode")
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self.filter_episodes)

        self.input_layout.addWidget(self.start_input)
        self.input_layout.addWidget(self.end_input)
        self.input_layout.addWidget(self.filter_button)
        self.layout.addLayout(self.input_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)

        self.setLayout(self.layout)
        self.display_episodes(self.episodes)

    def filter_episodes(self):
        start = int(self.start_input.text()) if self.start_input.text().isdigit() else None
        end = int(self.end_input.text()) if self.end_input.text().isdigit() else None

        if start is not None and end is not None:
            filtered_episodes = {k: v for k, v in self.episodes.items() if start <= k <= end}
            self.display_episodes(filtered_episodes)

    def display_episodes(self, episodes):

        for i in reversed(range(self.scroll_layout.count())):  # Clear previous content
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        print("Rebuilding the board...")
        for episode, data in episodes.items():
            board, last_action = build_board(data["actions"], data["best_q_values"], data["mcts_actions"])

            winner_text = "Player 1" if data["winner"] == 1 else "Player 2"
            episode_label = QLabel(f"Episode {episode} - Winner: {winner_text}, Total Turns: {data['turns']}")
            episode_label.setStyleSheet("font-size: 18px; font-weight: bold;")
            self.scroll_layout.addWidget(episode_label)

            grid_layout = QGridLayout()

            last_row, last_col, last_player = last_action if last_action else (None, None, None)

            for r in range(6):
                for c in range(7):
                    cell = board[r, c]
                    color = "white"

                    if cell == 'X':
                        color = "red"
                    elif cell == 'O':
                        color = "blue"
                    elif cell == '#':
                        color = "red"
                    elif cell == '$':
                        color = "blue"
                    elif cell == '@':
                        color = "blue"

                    # Highlight the last winning column
                    if last_row == r and last_col == c:
                        color = "pink" if last_player == 'X' else "lightblue"

                    label = QLabel(cell)
                    label.setAlignment(Qt.AlignCenter)
                    label.setStyleSheet(f"background-color: {color}; border: 1px solid black; width: 30px; height: 30px;")
                    grid_layout.addWidget(label, r, c)
            
            grid_widget = QWidget()
            grid_widget.setLayout(grid_layout)
            self.scroll_layout.addWidget(grid_widget)

if __name__ == "__main__":
    import sys

    log_file = "train.log"
    episodes = parse_log_file(log_file)

    app = QApplication(sys.argv)
    viewer = Connect4Viewer(episodes)
    viewer.show()
    sys.exit(app.exec_())
