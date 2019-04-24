from unittest import TestCase

from RecallPrecision import RecallPrecision


class TestRecallPrecision(TestCase):
    def setUp(self):
        self.rp = RecallPrecision('test')

    def test_plot(self):
        self.rp.addTP('Letters')
        self.rp.addTP('Orders')
        self.rp.addFP('Borders')
        self.rp.addTP('or')
        self.rp.addFP('tor')
        self.rp.addFN('Corners')
        self.rp.addFN('lore')
        self.rp.plot()
        # self.fail()
