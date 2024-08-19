from langchain_community.document_loaders import TextLoader, WebBaseLoader



def from_mock_data():
    
    loader = TextLoader("./data_source.txt")

    # docs = loader.load()
    
    return loader