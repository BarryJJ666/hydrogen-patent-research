// 诉讼类型节点
MERGE (lt:LitigationType {name: '侵权案件'});
MERGE (lt:LitigationType {name: '侵权案件, 其他案件, 权属案件'});
MERGE (lt:LitigationType {name: '其他案件'});
MERGE (lt:LitigationType {name: '其他案件, 无效诉讼'});
MERGE (lt:LitigationType {name: '无效诉讼'});
MERGE (lt:LitigationType {name: '无效诉讼, 其他案件'});
MERGE (lt:LitigationType {name: '权属案件'});
MERGE (lt:LitigationType {name: '权属案件, 其他案件'});
MERGE (lt:LitigationType {name: '行政案件'});
MERGE (lt:LitigationType {name: '行政案件, 其他案件'});
MERGE (lt:LitigationType {name: '行政案件, 无效诉讼'});