from socsim.bench.providers import DryadClient, DataverseClient, OSFClient

def test_imports():
    assert DryadClient and DataverseClient and OSFClient
