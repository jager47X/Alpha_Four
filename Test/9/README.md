#Change from 8
#changed dynamic reward system
                # if Model wins, then return high reward 5.0
                # if Model loses, then return high penality -5.0
                # if Model blocks the Trainer then return mid reward 1.8
                # if Model add 3 in row then return mid reward 1.3
                # if Trainer blocks the Model then return mid reward 0.9
                # if Trainer add 3 in row then return mid reward -1.3
                # if Model blocks 2 blocks in row then return mid reward  0.8
                # if Model add 2 in row then return mid reward 0.3
                # Otherwise -0.01

                # Calcurate trainer move as passive + model move 