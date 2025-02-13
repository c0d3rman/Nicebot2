import time

import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import nn

# Win reward
REWARD_WIN = 1.
# Draw reward
REWARD_DRAW = 0.
# Ordinary action reward
REWARD_ACTION = 0.

# Hidden layer size
hidden_layer_size = 50

# Reward discount rate
gamma = 0.8

# Initial exploration rate
epsilon_initial = 1.0
# Final exploration rate
epsilon_final = .01
# Number of training episodes to anneal epsilon
epsilon_anneal_episodes = 5000

# Learning rate
learning_rate = .001

# Number of training episodes to run
episode_max = 10000

# Number of training episodes to accumulate stats
episode_stats = 100

# Run name for tensorboard
run_name = "%s" % int(time.time())

# Directory for storing tensorboard summaries
summary_dir = '/tmp/tensorflow/tictactoe'


def train(session, graph_ops, summary_ops, saver):
    """
    Train model.
    """
    # Initialize variables
    session.run(tf.global_variables_initializer())

    # Initialize summaries writer for tensorflow
    writer = tf.summary.FileWriter(summary_dir + "/" + run_name, session.graph)
    summary_op = tf.summary.merge_all()

    # Unpack graph ops
    q_nn, q_nn_update, s, a, y, loss = graph_ops

    # Unpack summary ops
    win_rate_summary, episode_length_summary, epsilon_summary, loss_summary = summary_ops

    # Setup exploration rate parameters
    epsilon = epsilon_initial
    epsilon_step = (epsilon_initial - epsilon_final) / epsilon_anneal_episodes

    # X player state
    sx_t = np.empty([board_size, board_size], dtype=np.bool)
    # O player state
    so_t = np.empty_like(sx_t)

    # Accumulated stats
    stats = []

    # X move first
    move_x = True

    episode_num = 1

    while episode_num <= episode_max:
        # Start new game training episode
        sx_t[:] = False
        so_t[:] = False

        sar_prev = [(None, None, None), (None, None, None)]  # [(s, a, r(a)), (s(a), o, r(o)]

        move_num = 1

        while True:
            # Observe the next state
            s_t = create_state(move_x, sx_t, so_t)
            # Get Q values for all actions
            q_t = q_values(session, q_nn, s, s_t)
            # Choose action based on epsilon-greedy policy
            q_max_index, a_t_index = choose_action(q_t, sx_t, so_t, epsilon)

            # Retrieve previous player state/action/reward (if present)
            s_t_prev, a_t_prev, r_t_prev = sar_prev.pop(0)

            if s_t_prev is not None:
                # Calculate updated Q value
                y_t_prev = r_t_prev + gamma * q_t[q_max_index]
                # Apply equivalent transforms
                s_t_prev, a_t_prev = apply_transforms(s_t_prev, a_t_prev)
                # Update Q network
                q_update(session, q_nn_update,
                         s, s_t_prev,
                         a, a_t_prev,
                         y, [y_t_prev] * len(s_t_prev))

            # Apply action to state
            r_t, sx_t, so_t, terminal = apply_action(move_x, sx_t, so_t, a_t_index)

            a_t = np.zeros_like(sx_t, dtype=np.float32)
            a_t[a_t_index] = 1.

            if terminal:  # win or draw
                y_t = r_t  # reward for current player
                s_t_prev, a_t_prev, r_t_prev = sar_prev[-1]  # previous opponent state/action/reward
                y_t_prev = r_t_prev - gamma * r_t  # discounted negative reward for opponent

                # Apply equivalent transforms
                s_t, a_t = apply_transforms(s_t, a_t)
                s_t_prev, a_t_prev = apply_transforms(s_t_prev, a_t_prev)

                # Update Q network
                s_up = s_t + s_t_prev
                a_up = a_t + a_t_prev
                y_up = [y_t] * len(s_t) + [y_t_prev] * len(s_t_prev)
                q_update(session, q_nn_update, s, s_up, a, a_up, y, y_up)

                # Get episode loss
                loss_ep = q_loss(session, loss, s, s_up, a, a_up, y, y_up)

                # Play test game before next episode
                length_ep, win_x, win_o = test(session, q_nn, s)
                stats.append([win_x or win_o, length_ep, loss_ep])
                break

            # Store state, action and its reward
            sar_prev.append((s_t, a_t, r_t))

            # Next move
            move_x = not move_x
            move_num += 1

        # Scale down epsilon after episode
        if epsilon > epsilon_final:
            epsilon -= epsilon_step

        # Process stats
        if len(stats) >= episode_stats:
            mean_win_rate, mean_length, mean_loss = np.mean(stats, axis=0)
            print("episode: %d," % episode_num, "epsilon: %.5f," % epsilon,
                  "mean win rate: %.3f," % mean_win_rate, "mean length: %.3f," % mean_length,
                  "mean loss: %.3f" % mean_loss)
            summary_str = session.run(summary_op, feed_dict={win_rate_summary: mean_win_rate,
                                                             episode_length_summary: mean_length,
                                                             epsilon_summary: epsilon,
                                                             loss_summary: mean_loss})
            writer.add_summary(summary_str, episode_num)
            stats = []

        # Next episode
        episode_num += 1

    test(session, q_nn, s, dump=True)


def test(session, q_nn, s, dump=False):
    """
    Play test game.
    """
    # X player state
    sx_t = np.zeros([board_size, board_size], dtype=np.bool)
    # O player state
    so_t = np.zeros_like(sx_t)

    move_x = True
    move_num = 1

    if dump:
        print()

    while True:
        # Choose action
        s_t = create_state(move_x, sx_t, so_t)
        # Get Q values for all actions
        q_t = q_values(session, q_nn, s, s_t)
        _q_max_index, a_t_index = choose_action(q_t, sx_t, so_t, -1.)

        # Apply action to state
        r_t, sx_t, so_t, terminal = apply_action(move_x, sx_t, so_t, a_t_index)

        if dump:
            if terminal:
                if move_x:
                    _win, win_indices = check_win(sx_t)
                else:
                    _win, win_indices = check_win(so_t)
            else:
                win_indices = None
            print(Fore.CYAN + "Move:", move_num, Fore.RESET + "\n")
            dump_board(sx_t, so_t, a_t_index, win_indices, q_t)

        if terminal:
            if not r_t:
                # Draw
                if dump:
                    print("Draw!\n")
                return move_num, False, False
            elif move_x:
                # X wins
                if dump:
                    print("X wins!\n")
                return move_num, True, False
            # O wins
            if dump:
                print("O wins!\n")
            return move_num, False, True

        move_x = not move_x
        move_num += 1


def apply_transforms(s, a):
    """
    Apply state/action equivalent transforms (rotations/flips).
    """
    # Get composite state and apply action to it (with reverse sign to distinct from existing marks)
    sa = np.sum(s, 0) - a

    # Transpose state from [channel, height, width] to [height, width, channel]
    s = np.transpose(s, [1, 2, 0])

    s_trans = [s]
    a_trans = [a]
    sa_trans = [sa]

    # Apply rotations
    sa_next = sa
    for i in xrange(1, 4):  # rotate to 90, 180, 270 degrees
        sa_next = np.rot90(sa_next)
        if same_states(sa_trans, sa_next):
            # Skip rotated state matching state already contained in list
            continue
        s_trans.append(np.rot90(s, i))
        a_trans.append(np.rot90(a, i))
        sa_trans.append(sa_next)

    # Apply flips
    sa_next = np.fliplr(sa)
    if not same_states(sa_trans, sa_next):
        s_trans.append(np.fliplr(s))
        a_trans.append(np.fliplr(a))
        sa_trans.append(sa_next)
    sa_next = np.flipud(sa)
    if not same_states(sa_trans, sa_next):
        s_trans.append(np.flipud(s))
        a_trans.append(np.flipud(a))
        sa_trans.append(sa_next)

    return [np.transpose(s, [2, 0, 1]) for s in s_trans], a_trans


def same_states(s1, s2):
    """
    Check states s1 (or one of in case of array-like) and s2 are the same.
    """
    return np.any(np.isclose(np.mean(np.square(s1-s2), axis=(1, 2)), 0))


def create_state(move_x, sx, so):
    """
    Create full state from X and O states.
    """
    return np.array([sx, so] if move_x else [so, sx], dtype=np.float)


def choose_action(q, sx, so, epsilon):
    """
    Choose action index for given state.
    """
    # Get valid action indices
    a_vindices = np.where((sx+so)==False)
    a_tvindices = np.transpose(a_vindices)

    q_max_index = tuple(a_tvindices[np.argmax(q[a_vindices])])

    # Choose next action based on epsilon-greedy policy
    if np.random.random() <= epsilon:
        # Choose random action from list of valid actions
        a_index = tuple(a_tvindices[np.random.randint(len(a_tvindices))])
    else:
        # Choose valid action w/ max Q
        a_index = q_max_index

    return q_max_index, a_index


def apply_action(move_x, sx, so, a_index):
    """
    Apply action to state, get reward and check for terminal state.
    """
    if move_x:
        s = sx
    else:
        s = so
    s[a_index] = True
    win, _win_indices = check_win(s)
    if win:
        return REWARD_WIN, sx, so, True
    if check_draw(sx, so):
        return REWARD_DRAW, sx, so, True
    return REWARD_ACTION, sx, so, False


def q_values(session, q_nn, s, s_t):
    """
    Get Q values for actions from network for given state.
    """
    return q_nn.eval(session=session, feed_dict={s: [s_t]})[0]


def q_update(session, q_nn_update, s, s_t, a, a_t, y, y_t):
    """
    Update Q network with (s, a, y) values.
    """
    session.run(q_nn_update, feed_dict={s: s_t, a: a_t, y: y_t})


def q_loss(session, loss, s, s_t, a, a_t, y, y_t):
    """
    Get loss for (s, a, y) values.
    """
    return loss.eval(session=session, feed_dict={s: s_t, a: a_t, y: y_t})


def build_summaries():
    """
    Build tensorboard summaries.
    """
    win_rate_op = tf.Variable(0.)
    tf.summary.scalar("Win Rate", win_rate_op)
    episode_length_op = tf.Variable(0.)
    tf.summary.scalar("Episode Length", episode_length_op)
    epsilon_op = tf.Variable(0.)
    tf.summary.scalar("Epsilon", epsilon_op)
    loss_op = tf.Variable(0.)
    tf.summary.scalar("Loss", loss_op)
    return win_rate_op, episode_length_op, epsilon_op, loss_op


def build_graph():
    """
    Build tensorflow Q network graph.
    """
    s = tf.placeholder(tf.float32, [None, 2, board_size, board_size], name="s")

    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tf.transpose(s, [0, 2, 3, 1])

    # Flatten inputs
    net = tf.reshape(net, [-1, int(np.prod(net.get_shape().as_list()[1:]))])

    # Hidden fully connected layer
    net = layers.fully_connected(net, hidden_layer_size, activation_fn=nn.relu)

    # Output layer
    net = layers.fully_connected(net, board_size*board_size, activation_fn=None)

    # Reshape output to board actions
    q_nn = tf.reshape(net, [-1, board_size, board_size])

    # Define loss and gradient update ops
    a = tf.placeholder(tf.float32, [None, board_size, board_size], name="a")
    y = tf.placeholder(tf.float32, [None], name="y")
    action_q_values = tf.reduce_sum(tf.multiply(q_nn, a), axis=[1, 2])
    loss = tf.reduce_mean(tf.square(y - action_q_values))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    q_nn_update = optimizer.minimize(loss, var_list=tf.trainable_variables())

    return q_nn, q_nn_update, s, a, y, loss


def main(_):
    with tf.Session() as session:
        graph_ops = build_graph()
        summary_ops = build_summaries()
        saver = tf.train.Saver(max_to_keep=5)
        train(session, graph_ops, summary_ops, saver)

if __name__ == "__main__":
    tf.app.run()