#!/usr/bin/env python
from collections import namedtuple
import unittest

from logic.add_css_class import AddCSSClass
from tests.add_css_class.data.replace_matches_with_class import config


def assert_equal_test_template(*args):
    def test_template(self):

        return self.assert_response_equal(*args)
    return test_template


class TestReplaceMatchesWithClass(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def assert_response_equal(self, test_config):
        # Call method
        user_can_view_responses = AddCSSClass.replace_matches_with_class(**test_config.method_args)

        # Make assertion
        self.assertEqual(user_can_view_responses, test_config.response_val)


def setup_module():
    """This setup module is responsible for looking in the config associated with AddCSSClass.replace_matches_with_class
    and dynamically creating a test case for every set of arguments and responses associated with the config.

    :return:
    """
    TestConfig = namedtuple('TestConfig', ['response_val', 'method_args'])

    for behavior, update_data_test_case_data in config.items():
        test_config = TestConfig._make(update_data_test_case_data)

        test_name = "test_TestReplaceMatchesWithClass_replace_matches_with_class_{}_returns_{}".format(behavior, test_config.response_val)

        test_case = assert_equal_test_template(test_config)
        setattr(TestReplaceMatchesWithClass, test_name, test_case)


setup_module()
