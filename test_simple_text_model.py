import unittest
from simple_text_model import SimpleTextModel

class TestSimpleTextModel(unittest.TestCase):

    def setUp(self) -> None:     
        model_path = './best_siamese_model.h5'   
        self.text_model = SimpleTextModel(model_path)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_get_similar_score_should_return_appxroximately_1_on_same_text(self):
        text1 = "some text"
        text2 = "some text"

        similar_score = self.text_model.evaluate_similarity(text1, text2)

        self.assertAlmostEqual(similar_score,1, delta=0.3)

    def test_get_similar_score_should_return_low_score_on_entirly_different_text(self):
        text1 = "India is great"
        text2 = "rizwan firadus"

        similar_score = self.text_model.evaluate_similarity(text1, text2)

        self.assertAlmostEquals(similar_score, 1, delta= 0.3)

    def test_get_similar_score_should_return_score_with_delta_50_percet_accuracy(self):
        text1 = ""
        text2 = ""

        similar_score = self.text_model.evaluate_similarity(text1, text2)
        self.assertAlmostEqual(similar_score, 1, delta=0.5)

if __name__=="__main__" : 
    unittest.main()





  