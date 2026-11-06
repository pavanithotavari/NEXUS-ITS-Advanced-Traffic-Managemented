import numpy as np
from edge_traffic.deep_rl import ReplayBuffer, NeuralNetwork, DeepQLearningController


def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(capacity=50)
    s = np.zeros(20, dtype=np.float32)
    s_next = np.ones(20, dtype=np.float32)
    for i in range(40):
        buf.push(s, 1, 0.5, s_next, False)

    assert len(buf) == 40
    states, actions, rewards, next_states, dones = buf.sample(16)
    assert states.shape == (16, 20)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_states.shape == (16, 20)
    assert dones.shape == (16,)


def test_neural_network_forward_and_train():
    net = NeuralNetwork(in_dim=10, hidden_dim=16, out_dim=2, lr=0.01)
    x = np.random.randn(8, 10).astype(np.float32)
    y = np.random.randn(8, 2).astype(np.float32)

    initial_out = net.forward(x)
    assert initial_out.shape == (8, 2)

    loss_1 = net.train_step(x, y)
    loss_2 = net.train_step(x, y)
    assert isinstance(loss_1, float)
    assert loss_1 >= 0.0


def test_deep_q_controller_act_and_encode():
    ctrl = DeepQLearningController()
    queues = [5, 3, 8, 2]
    phase = 0
    elapsed = 4

    vec = ctrl.encode_state(queues, phase, elapsed, weather="rain")
    assert vec.shape == (20,)

    action = ctrl.act(queues, phase, elapsed)
    assert action in (0, 1)


def test_deep_q_controller_save_load(tmp_path):
    ctrl = DeepQLearningController()
    save_file = str(tmp_path / "test_dqn.npz")
    ctrl.save(save_file)

    ctrl2 = DeepQLearningController()
    ctrl2.load(save_file)
    np.testing.assert_allclose(ctrl.policy_net.W1, ctrl2.policy_net.W1)
