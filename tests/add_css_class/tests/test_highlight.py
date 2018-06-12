import unittest

from logic.add_css_class import AddCSSClass


class TestHighlight(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_wraps_sent_in_highlight(self):
        val = AddCSSClass.highlight("test sentence")
        self.assertEqual(val, "<span class='highlight'>test sentence</span>")
