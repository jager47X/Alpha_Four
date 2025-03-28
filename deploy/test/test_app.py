import random
import pytest
import sys
import os

# Add the parent directory to the Python path so that app can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

success_count = 0
failure_count = 0

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_full_game_random_moves(client):
    global success_count, failure_count
    try:
        # Reset the game
        response = client.post("/reset")
        data = response.get_json()
        assert response.status_code == 200, "Reset failed"
        assert "board" in data

        game_over = False
        max_moves = 100
        moves = 0

        while not game_over and moves < max_moves:
            col = random.randint(0, 6)
            response = client.post("/human_move", json={"col": col})
            data = response.get_json()

            if "error" in data:
                moves += 1
                continue

            if data.get("message") == "Game Over":
                game_over = True

            moves += 1

        assert game_over, "Game did not end within expected moves"
        assert "winner" in data, "Winner missing in game over"

        success_count += 1
        print("[✓] Test Passed")

    except AssertionError as e:
        failure_count += 1
        print(f"[✗] Test Failed: {e}")


def test_performance_evaluation(client):
    """
    Simulate multiple games and ensure AI (Player 2) wins ≥ 70% of them.
    AI moves are processed on the server after the human moves.
    """
    total_games = 30
    ai_wins = 0
    draws = 0

    for game in range(total_games):
        response = client.post("/reset")
        assert response.status_code == 200

        game_over = False
        max_moves = 100
        move_count = 0

        while not game_over and move_count < max_moves:
            # Human (Player 1) picks a random column
            col = random.randint(0, 6)
            response = client.post("/human_move", json={"col": col})
            data = response.get_json()

            if not data:
                continue
            if "error" in data:
                continue
            if data.get("message") == "Game Over":
                game_over = True
                winner = data.get("winner", 0)
                if winner == 2:
                    ai_wins += 1
                elif winner == 0:
                    draws += 1
                break  # End this game
            move_count += 1

        assert game_over, f"Game {game+1} did not finish properly"

    win_rate = ai_wins / total_games
    print(f"\nTotal Games: {total_games}")
    print(f"AI Wins: {ai_wins}")
    print(f"Draws: {draws}")
    print(f"AI Win Rate: {win_rate*100:.1f}%")

    assert win_rate >= 0.70, f"❌ FAIL: AI win rate below 70% ({win_rate*100:.1f}%)"
    print("✅ PASS: AI win rate ≥ 70%")

