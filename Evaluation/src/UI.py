
class Interface:
    # Game Renderer
    def render_board(env):
        plt.clf()
        board = np.flip(env.board, axis=0)
        rows, cols = board.shape
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        for row in range(rows):
            for col in range(cols):
                color = 'white'
                if board[row, col] == 1:
                    color = 'red'
                elif board[row, col] == 2:
                    color = 'yellow'
                circle = patches.Circle((col, row), 0.4, color=color, ec='black')
                ax.add_patch(circle)

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks([])
        ax.set_title("Connect 4")
        plt.pause(0.1)

    def main_human_vs_ai():
        global EPSILON, NUM_EPISODES

        env = Connect4()
        human_player = 1
        ai_player = 3 - human_player
        agent_logic = AgentLogic(policy_net=DQN().to(device))  # Use a pre-trained or new model

        plt.ion()  # Enable interactive mode for rendering

        while True:
            env.reset()
            done = False
            render_board(env)

            while not done:
                if env.current_player == human_player:
                    # Human's turn
                    valid_actions = env.get_valid_actions()
                    print(f"Valid actions: {valid_actions}")
                    try:
                        action = int(input("Enter the column number to drop your piece: "))
                        if action not in valid_actions:
                            raise ValueError("Invalid action.")
                    except ValueError as e:
                        print(e)
                        continue
                else:
                    # AI's turn
                    print(f"AI is thinking....")
                    action = agent_logic.logic_based_action(env,env.current_player,1)
                    if action is None:
                    action = agent_logic.combined_action(env)
                    print(f"AI chooses column: {action}")

                env.make_move(action, warning=1)
                render_board(env)

                # Check game status
                winner = env.check_winner()
                if winner:
                    print(f"Player {winner} wins!")
                    done = True
                elif env.is_draw():
                    print("It's a draw!")
                    done = True

            # Ask for replay
            render_board(env)
            replay = input("Play again? (y/n): ").strip().lower()
            if replay != 'y':
                break

        plt.ioff()  # Disable interactive mode
        print("Game over.")
