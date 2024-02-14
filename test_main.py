import unittest
import json
from fastapi.testclient import TestClient
from main import app
from simple_text_model import SimpleTextModel

class TestAPI(unittest.TestCase):

    def setUp(self) -> None:        
        self.client = TestClient(app)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_index_api(self):
        response = self.client.post("/", data='{"text1" : "India is great", "text2": "rizwan firadus"}')
        self.assertEqual(response.status_code, 200)
        response = response.json()
        self.assertAlmostEquals(response['similarity score'], 1, delta= 0.3)

if __name__=="__main__" : 
    unittest.main()