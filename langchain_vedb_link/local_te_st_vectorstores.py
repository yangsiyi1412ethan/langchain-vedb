import time
from typing import Generator

# import pytest
from fastapi.openapi.models import APIKey

from langchain_vedb_link.vectorstores import VeDB
from langchain_core.documents import Document
# from langchain_core.vectorstores import VectorStore
# from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_openai import OpenAIEmbeddings

APIKey = ""
embeddings = OpenAIEmbeddings(api_key=APIKey, model="text-embedding-3-small")

extra_column = {"year": "INT", "topic":"", "bar":"", "baz":""}

client_config = {"host":'xx.xx.xx.xx',  # 如：'192.168.1.88'
    "port":3306,
    "user":'vec_user',
    "password":'123456',              # 如果你授权时没设置密码，就留空
    "database":'vectorStore',     # 用你的实际数据库名
    "table_name":'langchain',  # 设置已有的表名或直接初始化一个表
    "charset":'utf8mb4',
    "metadata_columns": extra_column}
local_bytedance_store = VeDB(embedding=embeddings, client_config=client_config)
# # add_documents
document_1 = Document(id="1", page_content="foo", metadata={"year": "2025"})
document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
document_3 = Document(id="3", page_content="i will be deleted :(", metadata={"year": "2022", "topic": "magic"})
document_4 = Document(id="4", page_content="foo", metadata={"year": "2004", "topic": "magic"})
document_5 = Document(id="5", page_content="thudd", metadata={"bar": "baz"})
document_6 = Document(id="6", page_content="i will be deleteddd :(")
document_7 = Document(id="7", page_content="food", metadata={"baz": "bar"})
document_8 = Document(id="aa", page_content="thudt", metadata={"bar": "baz"})
document_9 = Document(id="cc", page_content="i will be deletedt :(")

#documents = [document_1, document_2, document_3]
documents = [document_1, document_2, document_3, document_4, document_5, document_6, document_7, document_8, document_9]
#documents = [document_4, document_5, document_6, document_7, document_8, document_9]
#local_bytedance_store.add_documents(documents=documents)

# delete by id
# local_bytedance_store.delete(ids=["cc"])

# get_by_ids
# results = local_bytedance_store.get_by_ids(["1", "2", "3"])
# for doc in results:
#     print(f"* {doc.id} {doc.page_content} [{doc.metadata}]")

# update
# updated_document_1 = Document(id="1", page_content="foott", metadata={"baz": "bar"})
# updated_document_2 = Document(id="2", page_content="thudbb", metadata={"bar": "baz"})
# updated_document_3 = Document(id="3", page_content="i will be deleted :(NEWa", metadata={"bar": "update"})
#
# documents = [updated_document_1, updated_document_2, updated_document_3]
# for doc in documents:
#     local_bytedance_store.update_document(document=doc)

# from_text
# docs = ["woo","duht","never deleted"]
# metadatas = [{},{"bar": "baz"},{"first": "have"}]
# local_bytedance_store = VeDB.from_texts(texts=docs, embedding=embeddings, metadatas=metadatas, client_config=client_config)

#similarity_search
# query = "thud"
# results = local_bytedance_store.similarity_search(query, k=10)
# for doc in results:
#     print(f"* {doc.page_content} [{doc.metadata}]")

# similarity_search_with_score
query = "never deleted"
results = local_bytedance_store.similarity_search_with_score(query, k=10)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

# similarity_search_with_filter
# query = "never deleted"
# results = local_bytedance_store.similarity_search(query, k=2, filter={
#      "topic": {"$eq": "magic"}
#     })
# results = local_bytedance_store.similarity_search(query, k=10, filter={
#          "topic": {"$like": "an%"}
#     })
# for doc in results:
#     print(f"* {doc.page_content} [{doc.metadata}]")

# similarity_search_with_score_with_filter
# query = "never deleted"
# results = local_bytedance_store.similarity_search_with_score(query, k=10, filter={
#         "$or":[{"year": {"$exists": True}}, {"topic": {"$eq": "magic"}}]
#     })
# results = local_bytedance_store.similarity_search_with_score(query, k=10, filter={
#          "topic": {"$ilike": "an%"}
#     })
# for doc, score in results:
#     print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

# test_langchain
# original_documents = [
#             Document(page_content="foo", metadata={"id": 1}),
#             Document(page_content="bar", metadata={"id": 2}),
#         ]
# ids = local_bytedance_store.add_documents(original_documents)
# time.sleep(5)
# documents = local_bytedance_store.similarity_search("bar", k=2)
# assert documents == [
#     Document(page_content="bar", metadata={"id": 2}, id=2),
#     Document(page_content="foo", metadata={"id": 1}, id=1),
# ]

# local_bytedance_store.clear_all()
# Verify that the original document object does not get mutated!
# (e.g., an ID is added to the original document object)
# assert original_documents == [
#     Document(page_content="foo", metadata={"id": 1}),
#     Document(page_content="bar", metadata={"id": 2}),
# ]
# for doc in documents:
#     print(f"* {doc.page_content} [{doc.metadata}]")


# documents = [
#     Document(page_content="foo", metadata={"id": 1}),
#     Document(page_content="bar", metadata={"id": 2}),
# ]
#
# local_bytedance_store.add_documents(documents=documents, ids=["1", "2"])
#
# # Now over-write content of ID 1
# new_documents = [
#     Document(
#         page_content="new foo", metadata={"id": 1, "some_other_field": "foo"}
#     ),
# ]
#
# local_bytedance_store.add_documents(documents=new_documents, ids=["1"])
#
# # Check that the content has been updated
# documents =  local_bytedance_store.similarity_search("new foo", k=2)
# print(documents)
# assert documents == [
#     Document(
#         id="1",
#         page_content="new foo",
#         metadata={"id": 1, "some_other_field": "foo"},
#     ),
#     Document(id="2", page_content="bar", metadata={"id": 2}),
# ]