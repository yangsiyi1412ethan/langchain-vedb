from typing import Generator

import pytest
from langchain_vedb.vectorstores import VeDB
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_openai import OpenAIEmbeddings

class TestByteDanceVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        APIKey = ""
        embeddings = OpenAIEmbeddings(api_key=APIKey, model="text-embedding-3-small")
        """Get an empty vectorstore for unit tests."""
        extra_column = {"id":"INT", "some_other_field":""}
        client_config = {"host": '10.37.115.167',  # 如：'192.168.1.88'xx.xx.xx.xx
                         "port": 24311,
                         "user": 'vector_user',
                         "password": '123456',  # 如果你授权时没设置密码，就留空
                         "database": 'vectorStore',  # 用你的实际数据库名
                         "table_name": 'langchain_test',  # 设置已有的表名或直接初始化一个表simple_test
                         "dimension": 1536,
                         "charset": 'utf8mb4',
                         "metadata_columns": extra_column}
        # client_config = {"host": 'xx.xx.xx.xx',  # 如：'192.168.1.88'xx.xx.xx.xx
        #                  "port": 3306,
        #                  "user": 'vec_user',
        #                  "password": '123456',  # 如果你授权时没设置密码，就留空
        #                  "database": 'vectorStore',  # 用你的实际数据库名
        #                  "table_name": 'langchain',  # 设置已有的表名或直接初始化一个表simple_test
        #                  "charset": 'utf8mb4',
        #                  "metadata_columns": extra_column}
        store = VeDB(embeddings, client_config)
        # store = ByteDanceVectorStore(self.get_embeddings(), client_config)
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            store.clear_all()
            store.close()
            # cleanup operations, or deleting data
            pass
