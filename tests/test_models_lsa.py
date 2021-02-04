from polycraft_nov_det.models.lsa.LSA_cifar10_no_est import LSACIFAR10NoEst


# constructor tests
def test_constructor_no_est():
    LSACIFAR10NoEst((3, 32, 32), 64)
    assert True
