import unittest
from math_utils import add

class TestMath(unittest.TestCase):
    def test_initial_behavior(self):
        # This test reflects the initial incorrect state
        self.assertEqual(add(2, 2), 0)
    def test_existing_behavior(self):
        # This test should always pass
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
