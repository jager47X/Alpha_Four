import os
import threading
import logging
import torch
import warnings
from flask import Flask, render_template, request, jsonify
from numba.core.errors import NumbaPerformanceWarning
import numpy as np
import webbrowser
import tkinter as tk
from PIL import Image, ImageTk

# Import your dependencies
from dependencies.environment import Connect4

from dependencies.agent import AgentLogic
from dependencies.utils import setup_logger, safe_make_dir, get_next_index
from dependencies.mcts import MCTS
# ----------------- Logging Setup ----------------- #
logging.basicConfig(
    filemode="w",       # Overwrite each run; change to 'a' to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Logging setup complete!")

# ----------------- Initialization ----------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: " + str(device))

# Compute the base directory (project root) based on the location of this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model based on a model version (from an environment variable or default)
model_version = 55
MODEL_PATH = os.path.join(BASE_DIR, "dependencies", "model", "Connect4_Agent_Model.pth")
logger.info(f"Loading Main DQN Agent from: {MODEL_PATH}")
if model_version>=45:
   from dependencies.layer_models.model2 import DQN
else:
   from dependencies.layer_models.model1 import DQN

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
policy_net = DQN().to(device)
policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
ai_agent = AgentLogic(policy_net, device)

# Create a global game state with a threading lock for concurrency
game_lock = threading.Lock()
game_env = Connect4()

# Configure Flask to use templates from the dependencies folder.
TEMPLATE_DIR = os.path.join(BASE_DIR, "dependencies", "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

def get_board_state():
    """Return the current board as a list of lists with native Python ints."""
    board_list = game_env.board.tolist()
    return [[int(cell) for cell in row] for row in board_list]

def check_game_over():
    winner = game_env.check_winner()
    return (winner != 0) or game_env.is_draw()

# ----------------- Splash Screen Function ----------------- #
def show_splash_screen(duration=3000):
    """
    Displays a splash screen for the given duration (in milliseconds)
    and shows the model version.
    """
    splash = tk.Tk()
    splash.overrideredirect(True)  # Remove window borders

    splash_image_path = os.path.join(BASE_DIR, "dependencies", "templates", "loading.png")
    image = Image.open(splash_image_path)
    photo = ImageTk.PhotoImage(image)
    
    # Display the splash image
    label_img = tk.Label(splash, image=photo)
    label_img.image = photo  # Keep a reference to avoid garbage collection
    label_img.pack()
    
    # Display the model version below the image
    version_text = f"v0.{model_version}"
    label_version = tk.Label(splash, text=version_text, font=("Helvetica", 16, "bold"),
                             bg="white", fg="black")
    label_version.pack()
    
    # Calculate geometry: add extra height for the version label (approx. 30 pixels)
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    img_width, img_height = image.size
    total_height = img_height + 30  # image height plus extra for version label
    x = (screen_width - img_width) // 2
    y = (screen_height - total_height) // 2
    splash.geometry(f"{img_width}x{total_height}+{x}+{y}")

    splash.after(duration, splash.destroy)
    splash.mainloop()

@app.route("/")
def index():
    board = get_board_state()
    return render_template("play.html", board=board)

@app.route("/reset", methods=["POST"])
def reset():
    with game_lock:
        game_env.reset()
        logger.info("Game reset")
    return jsonify({"status": "reset", "board": get_board_state()})

@app.route("/human_move", methods=["POST"])
def human_move():
    data = request.get_json()
    col = data.get("col")
    response = {}

    with game_lock:
        if game_env.current_player != 1:
            response["error"] = "Not your turn"
            return jsonify(response)
        success = game_env.make_move(col)
        if not success:
            response["error"] = "Invalid move"
            return jsonify(response)
        logger.info(f"Human -> Column {col+1}")
        if check_game_over():
            winner = int(game_env.check_winner())
            response["message"] = "Game Over"
            response["winner"] = winner
            response["board"] = get_board_state()
            return jsonify(response)

    def ai_move_thread():
        with game_lock:
            (model_used, dqn_action, mcts_action, hybrid_action, 
             best_q_val, mcts_value, hybrid_value, rand_action) = ai_agent.pick_action(
                game_env, epsilon=0.1, logger=logger, debug=True
            )
            if model_used == "mcts" and mcts_action is not None:
                action = mcts_action
            elif model_used == "random" and rand_action is not None:
                action = rand_action
            elif model_used == "dqn" and dqn_action is not None:
                action = dqn_action
            elif model_used == "hybrid" and hybrid_action is not None:
                action = hybrid_action
            else:
                action = None
            if action is not None:
                game_env.make_move(action)
                logger.info(
                    f"AI -> Column {action+1} (Q-Value: {best_q_val:.3f}) "
                    f"(mcts-Value: {mcts_value:.3f}) (hybrid-Value: {hybrid_value:.3f})"
                )
    t = threading.Thread(target=ai_move_thread)
    t.start()
    t.join()

    with game_lock:
        if check_game_over():
            winner = int(game_env.check_winner())
            response["message"] = "Game Over"
            response["winner"] = winner
        else:
            response["message"] = "Move accepted"
        response["board"] = get_board_state()

    return jsonify(response)

if __name__ == "__main__":
    show_splash_screen(duration=3000)  # Display the splash screen for 3 seconds
    threading.Thread(target=open_browser).start()
    app.run(debug=True, use_reloader=False)
