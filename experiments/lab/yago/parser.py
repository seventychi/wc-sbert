from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

# 创建一个RDF图
g = Graph()

# 从YAGO文件加载数据
g.parse("../../data/yago-4.5.0.2/yago-facts.ttl", format="turtle", encoding="utf-8")

print(f"Graph has {len(g)} facts.")

# 定义SPARQL查询
query = prepareQuery(
    """
    SELECT ?subject ?predicate ?object
    WHERE {
        ?subject ?predicate ?object
    }
    """,
    initNs={"rdf": g.namespace("rdf"), "rdfs": g.namespace("rdfs")}
)

# 执行查询并打印结果
for row in g.query(query):
    subject, predicate, obj = row
    print(f"Subject: {subject}, Predicate: {predicate}, Object: {obj}")