from tutorials.torch import simple_nn


def test_import_nn():
    my_nn = simple_nn.NN(1, 1)
    print(my_nn._get_name())
    print("Load: OK")
