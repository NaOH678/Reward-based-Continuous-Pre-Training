import torch
import unittest
from flame.models.future_encoder import FutureEncoder
from flame.models.mi_estimator import build_mi_estimator

class TestFutureEncoderAndMIEstimator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 32
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

    def test_future_encoder_output_shape(self):
        encoder = FutureEncoder(self.hidden_size, future_k=4)
        summaries = encoder(self.hidden_states)
        self.assertEqual(summaries.shape, self.hidden_states.shape)

    def test_mi_estimator_infonce(self):
        encoder = FutureEncoder(self.hidden_size, future_k=2, summary_method="mean")
        estimator = build_mi_estimator("infonce", self.hidden_size)
        
        summaries = encoder(self.hidden_states)
        loss = estimator(self.hidden_states, summaries)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)

    def test_mi_estimator_backward(self):
        encoder = FutureEncoder(self.hidden_size, future_k=2)
        estimator = build_mi_estimator("infonce", self.hidden_size)
        
        self.hidden_states.requires_grad = True
        summaries = encoder(self.hidden_states)
        loss = estimator(self.hidden_states, summaries)
        loss.backward()
        
        self.assertIsNotNone(self.hidden_states.grad)

if __name__ == '__main__':
    unittest.main()
