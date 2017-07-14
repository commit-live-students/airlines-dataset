from unittest import TestCase
from sklearn import linear_model

class TestImport_data(TestCase):
    def test_import_data(self):
        from build import import_data
        columns = [u'type_traveller', u'cabin_flown',
                   u'overall_rating', u'seat_comfort_rating',
                   u'cabin_staff_rating', u'food_beverages_rating',
                   u'inflight_entertainment_rating', u'ground_service_rating',
                   u'wifi_connectivity_rating', u'value_money_rating']

        encoding = {"cabin_flown": ["Economy", "Premium Economy", "Business Class", "First Class"]}

        df, dropped = import_data("./data/airline.csv", columns, encoding, 0.2)
        self.assertTrue("type_traveller" in dropped)
        self.assertTrue("inflight_entertainment_rating" in dropped)
        self.assertTrue("ground_service_rating" in dropped)
        self.assertTrue("wifi_connectivity_rating" in dropped)


    def test_model_tuning(self):
        from build import import_data, model_tuning
        columns = [u'type_traveller', u'cabin_flown',
                   u'overall_rating', u'seat_comfort_rating',
                   u'cabin_staff_rating', u'food_beverages_rating',
                   u'inflight_entertainment_rating', u'ground_service_rating',
                   u'wifi_connectivity_rating', u'value_money_rating']

        encoding = {"cabin_flown": ["Economy", "Premium Economy", "Business Class", "First Class"]}

        df, dropped = import_data("./data/airline.csv", columns, encoding, 0.2)

        predictors = ['cabin_flown', 'seat_comfort_rating', 'cabin_staff_rating', 'food_beverages_rating',
                      'value_money_rating']
        target = 'overall_rating'
        best_score_, best_model = model_tuning(df, predictors, target)

        self.assertTrue(isinstance(best_model, linear_model.LinearRegression))