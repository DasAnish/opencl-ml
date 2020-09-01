from clobject import *
from pytest import raises


def test_singleton_implementation():
    # making first instance
    clsingleton_1 = ClSingleton()
    clsingleton_2 = ClSingleton.get_instance()
    code_1 = Code()
    code_2 = Code.get_instance()

    # testing if it is a singleton, making another instance
    with raises(Exception):
        ClSingleton()
        Code()

    # test that second intance has the same reference.
    assert clsingleton_1 == clsingleton_2
    assert code_1 == code_2

    # testing that members are the same
    assert clsingleton_2.context == clsingleton_1.context
    assert clsingleton_2.queue == clsingleton_1.queue

    assert code_1.program == code_2.program



